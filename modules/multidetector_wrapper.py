import mmdet.models
import mmdet.structures
import torch
import torch.nn as nn
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.registry import MMENGINE_MODELS, MMENGINE_TASK_UTILS
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import FasterRCNN, RTMDet, DETR
from mmengine.runner import load_checkpoint
from mmdet.registry import HOOKS as MMDET_HOOKS
from mmengine.registry import HOOKS as MMENGINE_HOOKS


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
    """
    Wraps 3 frozen detectors for multi-objective training.
    """
    def __init__(self,
                 preprocessor_cfg,
                 detector_cfgs,
                 detector_ckpts,
                 loss_weights=None,
                 data_preprocessor=None,   
                 init_cfg=None):            # <-- keep standard BaseDetector args
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.preprocessing = MODELS.build(preprocessor_cfg)
        if loss_weights is None:
            loss_weights = [1.0] * len(detector_cfgs)
        self.loss_weights = loss_weights
        detector_classes = {
            'FasterRCNN': FasterRCNN,
            'RTMDet': RTMDet,
            'DETR': DETR
        }

        detectors = []
        for cfg in detector_cfgs:
            cfg = cfg.copy()
            cls_name = cfg.pop('type')
            cls = detector_classes[cls_name]
            detectors.append(cls(**cfg))
        self.detectors = nn.ModuleList(detectors)
        for detector, ckpt_path in zip(self.detectors, detector_ckpts):
            load_checkpoint(detector, ckpt_path, map_location='cpu', strict=False)
        for detector in self.detectors:
            for param in detector.parameters():
                param.requires_grad = False
        
    
    def loss(self, batch_inputs, data_samples):
        preprocessed = self.preprocessing(batch_inputs) # step1: shared preprocessing
        #step2: forward through each detector 
        # Sequential for memory efficiency (reuse activation space)
        losses = {}
        for i, detector in enumerate(self.detectors):
            det_losses = detector.loss(preprocessed, data_samples)
            for k, v in det_losses.items():
                new_key = f'det{i}_{k}'
                if torch.is_tensor(v):
                    losses[new_key] = v * self.loss_weights[i]
                else:
                    # e.g. lists/dicts of tensors: keep as is
                    losses[new_key] = v
        return losses
        
        # Each detector returns a dict like {'loss_cls': 0.5, 'loss_bbox': 0.3}
        
    def predict(self, batch_inputs, data_samples):
        preprocessed = self.preprocessing(batch_inputs)
        return self.detectors[0].predict(preprocessed, data_samples)
    
    def extract_feat(self, batch_inputs):
        return self.preprocessing(batch_inputs)

    def _forward(self, batch_inputs, data_samples=None):
        return self.preprocessing(batch_inputs)