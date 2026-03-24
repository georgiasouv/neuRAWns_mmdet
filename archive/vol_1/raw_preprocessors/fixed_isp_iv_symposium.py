
import torch
import torch.nn as nn
from .base_preprocessor import BasePreprocessor
from mmdet.registry import MODELS

@MODELS.register_module()
class FixedISP(BasePreprocessor):
    """Fixed ISP pipeline with no learnable parameters - for validation only."""
    
    def __init__(self, norm_threshold=0.95, gamma=2.2):
        super().__init__()
        self.norm_threshold = norm_threshold
        self.gamma = gamma
        
        # ImageNet normalization constants (fixed, not learnable)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        

    def forward(self, x):
        """
        Args:
            x: Raw Bayer image, shape (B, 1, H, W) or (B, H, W)
        Returns:
            ImageNet-normalized RGB, shape (B, 3, H/2, W/2)
        """
        # Step 1: Packing (demosaicing via channel extraction)
        x = self.packing(x)  # (B, 3, H/2, W/2)
        # Step 2: Normalization (p99 clipping)
        x = self.adaptive_norm(x)  # (B, 3, H/2, W/2), range [0,1]
        x = self.awb(x)  # (B, 3, H/2, W/2), range [0,1]
        x = self.gamma_correction(x)  # (B, 3, H/2, W/2), range [0,1]
        # x = torch.clamp(x * 255.0, 0, 255.0) / 255.0
        # Step 5: ImageNet normalization
        # Ensure mean and std are on same device as x
        mean = self.mean.to(x.device) if self.mean.device != x.device else self.mean
        std = self.std.to(x.device) if self.std.device != x.device else self.std
        x = (x - mean) / std
            
        return x
    
    def packing(self, x):
        """
        Demosaic Bayer pattern by extracting and averaging G channels.
        RGGB pattern: R(0,0), G(0,1), G(1,0), B(1,1)
        Handles odd dimensions by cropping.
        """
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
        
        # Ensure even dimensions for proper Bayer packing
        B, H, W = x.shape
        H_even = (H // 2) * 2
        W_even = (W // 2) * 2
        
        if H != H_even or W != W_even:
            x = x[:, :H_even, :W_even]  # Crop to even dimensions
        
        # Extract channels
        R = x[:, 0::2, 0::2]   # Top-left
        G1 = x[:, 0::2, 1::2]  # Top-right
        G2 = x[:, 1::2, 0::2]  # Bottom-left
        B = x[:, 1::2, 1::2]   # Bottom-right
        
        # Average the two green channels
        G = (G1 + G2) * 0.5
        
        # Stack to RGB
        packed = torch.stack([R, G, B], dim=1)  # (B, 3, H/2, W/2)
        return packed
    
    def adaptive_norm(self, x):
        """Normalize using p99 quantile per channel."""
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, -1)  # (B, C, H*W)
        
        # Compute p99 per channel
        norm = torch.quantile(x_flat, self.norm_threshold, dim=-1, keepdim=True)  # (B, C, 1)
        norm = norm.unsqueeze(-1)  # (B, C, 1, 1)
        norm = torch.clamp(norm, min=1e-6)
        
        # Normalize and clip
        x = x / norm
        x = torch.clamp(x, 0, 1)
        return x
    
    def awb(self, x):
        mean_r = x[:, 0:1, :, :].mean(dim=(2, 3), keepdim=True)  # [4, 1, 1, 1]
        mean_g = x[:, 1:2, :, :].mean(dim=(2, 3), keepdim=True)  # [4, 1, 1, 1]
        mean_b = x[:, 2:3, :, :].mean(dim=(2, 3), keepdim=True)  # [4, 1, 1, 1]
        
        # Scale R and B to G
        x[:, 0:1] = x[:, 0:1] * (mean_g / (mean_r + 1e-6))
        x[:, 2:3] = x[:, 2:3] * (mean_g / (mean_b + 1e-6))
        
        # Clip after AWB
        x = torch.clamp(x, 0, 1)
        return x
    
    def gamma_correction(self, x):
        """Apply gamma correction."""
        return torch.pow(x, 1.0 / self.gamma)