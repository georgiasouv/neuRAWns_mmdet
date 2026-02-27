import torch 
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models import build_detector
from mmengine.runner import load_checkpoint


@MODELS.register_module()
class MultiDetectorModel(nn.Module):
    """
    Wraps 3 frozen detectors for multi-objective training.
    """
    def __init__(self,
                 preprocessor_cfg,
                 detector_cfgs,
                 detector_ckpts,
                 loss_weights=None):
        super().__init__()
        self.preprocessing = MODELS.build(preprocessor_cfg)
        if loss_weights is None:
            loss_weights = [1.0] * len(detector_cfgs)
            self.loss_weights = loss_weights
        self.detectors = nn.ModuleList([build_detector(cfg) for cfg in detector_cfgs])
        for detector, ckpt_path in zip(self.detectors, detector_ckpts):
            load_checkpoint(detector, ckpt_path, map_location='cpu')
        for detector in self.detectors:
            for param in detector.parameters():
                param.requires_grad = False
        
    
    def forward(self, batch_inputs, data_samples):
        preprocessed = self.preprocessing(batch_inputs) # step1: shared preprocessing
        #step2: forward through each detector 
        # Sequential for memory efficiency (reuse activation space)
        losses = {}
        for i, detector in enumerate(self.detectors):
            det_losses = detector.loss(preprocessed, data_samples)
            # Prefix keys to avoid collision: loss_cls → det0_loss_cls
            for k, v in det_losses.items():
                losses[f'det{i}_{k}'] = v * self.loss_weights[i]
        return losses
        
        # Each detector returns a dict like {'loss_cls': 0.5, 'loss_bbox': 0.3}
        
    def predict(self, batch_inputs, data_samples):
        preprocessed = self.preprocessing(batch_inputs)
        return self.detectors[0].predict(preprocessed, data_samples)