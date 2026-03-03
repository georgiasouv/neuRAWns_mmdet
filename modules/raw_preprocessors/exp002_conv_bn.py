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
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=True)
        self.activation = nn.LeakyReLU(0.2)
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        
        # Initialize conv to output balanced RGB channels
        with torch.no_grad():
            # Initialize weights small so outputs average the 4 RGGB channels
            nn.init.constant_(self.conv.weight, 0.01)
            # # Initialize bias to push outputs toward ImageNet means
            # self.conv.bias.copy_(torch.tensor([0.485, 0.456, 0.406]))

    def forward(self, x):
        # x is (B, 1, H, W) from DetDataPreprocessor
        B, C, H, W = x.shape

        x = self.packing(x)        # (B, 4, H/2, W/2)
        x = self.adaptive_norm(x)
        x = self.gamma(x)
        x = self.conv(x)           # (B, 3, H/2, W/2)
        x = self.activation(x)

        scale = torch.clamp(self.output_scale, min=0.1, max=10.0)
        x = x * scale
        x = torch.clamp(x, 0, 1)

        # ImageNet normalization
        # mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        # std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        # x = (x - mean) / std

        # NEW: upsample back to original padded size (H, W)
        x = torch.nn.functional.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=False
        )

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
        
        
        