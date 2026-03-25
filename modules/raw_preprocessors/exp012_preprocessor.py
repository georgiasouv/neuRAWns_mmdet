import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_preprocessor import BasePreprocessor
from mmdet.registry import MODELS


@MODELS.register_module()
class Exp012Processor(BasePreprocessor):  # safe, almost fixed, learnable preprocessor
    def __init__(self, norm_threshold=0.99, debug=False):
        super().__init__()
        self.norm_thr = norm_threshold
        self.debug = debug
        self.channel_gain = nn.Parameter(torch.ones(3))   # Learnable per-channel gain for white balance correction  || init=1 (identity), constrained to [0.5, 1.5] in forward
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True, groups=3) # Depthwise conv — spatial refinement, no colour mixing || groups=3 means each channel gets its own independent kernel
        self.activation = nn.LeakyReLU(0.1)
        with torch.no_grad():               # Identity initialisation — network starts as pure classical pipeline         
            nn.init.constant_(self.conv.weight, 0.0)
            self.conv.bias.zero_()


    def forward(self, x):
        # x: [B, 3, H, W] float32, range [0, 2^24]
        B, C, H, W = x.shape

        # 1) p99 global luminance normalisation — one scalar per image
        # flattening across all channels together gives one global scale factor
        flat = x.view(B, -1)
        q = torch.quantile(flat, self.norm_thr, dim=-1, keepdim=True)
        q = torch.clamp(q, min=1e-6).view(B, 1, 1, 1)
        x = torch.clamp(x / q, 0.0, 1.0)

        # 2) Gamma correction — perceptual linearisation
        # after this, x is [0,1] perceptually scaled
        x = x ** (1.0 / 2.2)

        # 3) Per-channel gain — learnable white balance
        # constrained to [0.5, 1.5] for physical plausibility and training stability
        gain = torch.clamp(self.channel_gain, 0.5, 1.5).view(1, 3, 1, 1)
        x = torch.clamp(x * gain, 0.0, 1.0)

        # 4) Depthwise conv as residual — learns spatial corrections
        # on top of classical output, not instead of it
        # if weights are zero: residual=0, x unchanged — training starts from classical
        residual = self.conv(x)
        residual = self.activation(residual)
        x = torch.clamp(x + residual, 0.0, 1.0)

        if self.debug:
            print(f"[DEBUG] gain: {self.channel_gain.data}")
            print(f"[DEBUG] residual range: [{residual.min():.4f}, {residual.max():.4f}]")
            print(f"[DEBUG] output range:   [{x.min():.4f}, {x.max():.4f}]")
            print(f"[DEBUG] ch_mean: {x.mean(dim=(0, 2, 3))}")

        return x