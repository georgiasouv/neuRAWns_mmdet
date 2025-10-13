import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.backbones import ResNet

@MODELS.register_module()
class RAWResNet(nn.Module):
    def __init__(self,
                 preprocess_cfg=dict(
                     type='RAWPreprocess_v1',
                     in_channels=4,
                     out_channels=3,
                     norm_threshold=0.99),
                 **resnet_kwargs):
        super().__init__()
        self.preprocessor = MODELS.build(preprocess_cfg)
        self.resnet = ResNet(**resnet_kwargs)
    
    def forward(self, x):
        # x.shape (B, 1, H, W) OR (B, H, W)
        x = self.preprocessor(x) # --> (B, 3, H/2, W/2)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) # --> (B, 3, H, W)
        out = self.resnet(x)
        return out
    
    def init_weights(self):
        self.resnet.init_weights()