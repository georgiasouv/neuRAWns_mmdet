import torch 
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models import build_detector

@MODELS.register_module()
class MultiDetectorModel(nn.Module):
    """
    Wraps 3 frozen detectors for multi-objective training.
    """
    def __init__(self, 
                 detector_cfgs,
                 detector_ckpts,
                 loss_weights=None):
        super().__init__()
        
        
        pass
    
    def forward(self, batch_inputs, data_samples):
        preprocessed = self.prepricessing(batch_inputs) # step1: shared preprocessing
        #step2: forward through each detector 
        # Sequential for memory efficiency (reuse activation space)
        losses = {}
        for i, detector in enumerate(self.detectors):
            det_losses = detector.loss(preprocessed, data_samples)
            # Prefix keys to avoid collision: loss_cls → det0_loss_cls
            for k, v in det_losses.items():
                losses[f'det{i}_{k}'] = v
        
        return losses
        
        # Each detector returns a dict like {'loss_cls': 0.5, 'loss_bbox': 0.3}
        
        