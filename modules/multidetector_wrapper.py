import mmdet.models
import mmdet.structures
from mmengine.structures import InstanceData
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
        
    
    def loss(self, batch_inputs, data_samples):   # USED DURING TRAINING 
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
        
    def predict(self, batch_inputs, data_samples, debug_all=False):     # USED DURING VALIDATION/ INFERENCE
       # 1) shared preprocessing: raw (N,1,H,W) -> rgb (N,3,H,W)
        preprocessed = self.preprocessing(batch_inputs)
        
        if not debug_all:
            # original behaviour: only detector 0 used for eval
            return self.detectors[0].predict(preprocessed, data_samples)

        # 2) run all detectors
        all_outs = []  # list[num_det][batch] of DetDataSample
        for det in self.detectors:
            outs = det.predict(preprocessed, data_samples)
            all_outs.append(outs)
            
        merged = []
        for img_idx, base_ds in enumerate(data_samples):
            # copy base sample (metainfo, gt_instances, etc.)
            ds = base_ds.new()
            
            # attach per detecotr predictions
            for det_idx, det_out in enumerate(all_outs):
                inst = det_out[img_idx].pred_instances  # InstanceData
                setattr(ds, f'det{det_idx}_pred_instances', inst)

            # for compatibility with evaluator, use detector 0 as main pred
            ds.pred_instances = getattr(ds, 'det0_pred_instances')

            merged.append(ds)

        return merged

            
    
    def extract_feat(self, batch_inputs):
        return self.preprocessing(batch_inputs)

    def _forward(self, batch_inputs, data_samples=None):
        return self.preprocessing(batch_inputs)
    
    
"""
    batch_inputs = the image tensor batch (what the network sees).
    batch_inputs.shape == (2, 1, 800, 1333)
    dim 0: batch dimension (2 images)
    dim 1: channels (1, Bayer)
    dim 2 - 3: height, width
    
    data_samples = a python list of per image objects with metadata and ground truth. So for batch_size =2 , len(data_samples) == 2
    
    data_samples[0] = DetDataSample(
        metainfo = {
            'img_id': 101,
            'img_path': 'cifs/Shares/WMG/Data/ROD/images/raw/train/000101.png',
            'ori_shape': (1080, 1920, 3),    # original H,W,C
            'img_shape': (800, 1333, 3),     # after Resize in pipeline
            'scale_factor': (0.74, 0.74, 0.74, 0.74),
            'flip': False,
            'flip_direction': None,
        },

        gt_instances = InstanceData(
            # 3 ground-truth boxes in this image
            bboxes = tensor([
                [ 150.3,  200.5,  400.7,  600.2],  # car
                [ 600.0,  220.0,  800.0,  700.0],  # truck
                [ 300.0,  500.0,  340.0,  620.0],  # person
            ]),   # shape (3, 4), xyxy in resized image coords
            labels = tensor([2, 4, 0]),  # indices into classes=('person','bicycle','car','train','truck')
        ),
        )

    # during training, pred_instances may be empty or used internally
    # during evaluation, model will add:
    # pred_instances = InstanceData(...)
)

...
    
    data_samples = [DetDataSample_for_img101, DetDataSample_for_img102]
    
    """
