import copy
import torch
import torch.nn as nn
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import FasterRCNN, RTMDet, DETR
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.registry import MMENGINE_MODELS, MMENGINE_TASK_UTILS
from mmdet.registry import HOOKS as MMDET_HOOKS
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.runner.checkpoint import _load_checkpoint, load_checkpoint


def sync_registry(mmdet_reg, mmengine_reg):
    for name, module in mmdet_reg._module_dict.items():
        if name not in mmengine_reg._module_dict:
            mmengine_reg._module_dict[name] = module


sync_registry(MODELS, MMENGINE_MODELS)
sync_registry(TASK_UTILS, MMENGINE_TASK_UTILS)
sync_registry(MMDET_HOOKS, MMENGINE_HOOKS)


@MMENGINE_MODELS.register_module()
@MODELS.register_module()
class MultiDetectorModel(BaseDetector):

    def __init__(self, preprocessor_cfg, detector_cfgs, detector_ckpts,
                 loss_weights=None, data_preprocessor=None, init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.preprocessor = MODELS.build(preprocessor_cfg)
        if loss_weights is None:
            loss_weights = [1.0] * len(detector_cfgs)
        self.loss_weights = loss_weights
        detector_classes = {'FasterRCNN': FasterRCNN, 'RTMDet': RTMDet, 'DETR': DETR}
        detectors, names = [], []
        for cfg in detector_cfgs:
            cfg = cfg.copy()
            cls_name = cfg.pop('type')
            detectors.append(detector_classes[cls_name](**cfg))
            base = cls_name.lower()
            n, name = base, base
            while name in names:
                n += 1
                name = f'{base}{n}'
            names.append(name)
        self.detectors = nn.ModuleList(detectors)
        self.detector_names = names
        self._load_and_verify_checkpoints(detector_ckpts)
        self._freeze_detectors()
        self.eval_detector = self.detector_names[0]
        self._last_norm_input = None
        self._last_enhanced = None
        self._last_det_input_stats = {}
        self._last_per_detector_grads = None

    def _load_and_verify_checkpoints(self, detector_ckpts):
        for det, name, path in zip(self.detectors, self.detector_names, detector_ckpts):
            ckpt = _load_checkpoint(path, map_location='cpu')
            state = ckpt.get('state_dict', ckpt)
            model_keys = set(det.state_dict().keys())
            coverage = len(model_keys & set(state.keys())) / max(len(model_keys), 1)
            load_checkpoint(det, path, map_location='cpu', strict=False)
            if coverage < 0.99:
                raise RuntimeError(f'[{name}] checkpoint {path} covers only {coverage:.1%} of model keys.')
            print(f'[{name}] checkpoint coverage {coverage:.1%} OK')

    def _freeze_detectors(self):
        for det in self.detectors:
            for p in det.parameters():
                p.requires_grad = False
            det.eval()

    def train(self, mode=True):
        super().train(mode)
        self.preprocessor.train(mode)
        for det in self.detectors:
            det.eval()
        return self

    def _enhance(self, batch_inputs):
        self._last_norm_input = batch_inputs.detach()
        enhanced = self.preprocessor(batch_inputs)
        self._last_enhanced = enhanced.detach()
        return enhanced

    def _detector_loss(self, name, det, enhanced, data_samples):
        data = {'inputs': enhanced, 'data_samples': copy.deepcopy(data_samples)}
        data = det.data_preprocessor(data, training=True)
        x = data['inputs']
        self._last_det_input_stats[name] = {
            'mean': x.mean().item(), 'std': x.std().item(),
            'min': x.min().item(), 'max': x.max().item()}
        return det.loss(x, data['data_samples'])

    def loss(self, batch_inputs, data_samples):
        enhanced = self._enhance(batch_inputs)
        losses = {}
        for i, (name, det) in enumerate(zip(self.detector_names, self.detectors)):
            det_losses = self._detector_loss(name, det, enhanced, data_samples)
            for k, v in det_losses.items():
                losses[f'{name}.{k}'] = v * self.loss_weights[i] if torch.is_tensor(v) else v
        return losses

    def predict(self, batch_inputs, data_samples, debug_all=False):
        enhanced = self._enhance(batch_inputs)

        def run(name):
            det = self.detectors[self.detector_names.index(name)]
            data = {'inputs': enhanced, 'data_samples': copy.deepcopy(data_samples)}
            data = det.data_preprocessor(data, training=False)
            return det.predict(data['inputs'], data['data_samples'], rescale=True)

        if not debug_all:
            return run(self.eval_detector)
        all_outs = [run(n) for n in self.detector_names]
        merged = []
        for img_idx, base_ds in enumerate(data_samples):
            ds = base_ds.new()
            for det_idx, det_out in enumerate(all_outs):
                setattr(ds, f'{self.detector_names[det_idx]}_pred_instances',
                        det_out[img_idx].pred_instances)
            ds.pred_instances = all_outs[0][img_idx].pred_instances
            merged.append(ds)
        return merged

    def extract_feat(self, batch_inputs):
        return self._enhance(batch_inputs)

    def _forward(self, batch_inputs, data_samples=None):
        return self._enhance(batch_inputs)
