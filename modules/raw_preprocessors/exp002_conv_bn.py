import torch
import torch.nn as nn
from .base_preprocessor import BasePreprocessor
from mmdet.registry import MODELS

@MODELS.register_module()
class Exp002ConvBN(BasePreprocessor):
    def __init__(self, in_channels=4, out_channels=3, norm_threshold=0.95):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_thr = norm_threshold
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.packing(x)
        x = self.adaptive_norm(x)
        x = self.gamma(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x
    

    def packing(self, x): # RGGB
        """
        Args x: Bayer image, shape (B, H, W) or (B, 1, H, W)                                                                                                                                                                                                                                        
        Returns packed: shape (B, 4, H/2, W/2)
        """
        if x.dim() == 4:
            x = x.squeeze(1) # (B, 1, H, W) -> (B, H, W)
        R = x[:, 0::2, 0::2]
        G1 = x[:, 0::2, 1::2]  
        G2 = x[:, 1::2, 0::2]  
        B = x[:, 1::2, 1::2]
        
        packed = torch.stack([R, G1, G2, B], dim=1)  # (B, 4, H/2, W/2)
        return packed
            
    def adaptive_norm(self, x):
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, -1)  # (B, C, H*W)
        norm = torch.quantile(x_flat, self.norm_thr, dim=-1, keepdim=True) # (B, C, 1) -- quantile along the flattened(most rightside) dimension
        norm = norm.unsqueeze(-1)   # (B, C, 1, 1) for broadcasting
        norm = torch.clamp(norm, min=1e-6)
        x = x / norm
        x = torch.clamp(x, 0, 1)
        return x
    
    
    def gamma(self, x):
        return x**(1/2.2)
        
        
        