from .base_preprocessor import BasePreprocessor
from .helper_functions import identity_conv_init, gamma_warmstart_deltas, apply_lut
from mmdet.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F


@MODELS.register_module()
class GuidedLocalToneMap(BasePreprocessor):
    """Global learnable curve + adaptive local contrast (the real HDR fix)."""
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 knots=16, blur_k=15):
        super().__init__()
        self.knots = knots
        self.log_gain = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.curve_deltas = nn.Parameter(gamma_warmstart_deltas(in_channels, knots))

        # fixed depthwise box blur → illumination estimate
        self.blur_k = blur_k
        w = torch.ones(in_channels, 1, blur_k, blur_k) / (blur_k * blur_k)
        self.register_buffer('blur_w', w)

        self.local_gain = nn.Parameter(torch.zeros(1))   # starts at 0 → passthrough
        self.eps = 1e-3
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2, bias=True)
        identity_conv_init(self.conv, in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = x * torch.exp(self.log_gain)
        base = apply_lut(x, self.curve_deltas, self.knots)
        illum = F.conv2d(base, self.blur_w, padding=self.blur_k // 2,
                         groups=base.shape[1])
        detail = base / (illum + self.eps)
        x = base + self.local_gain * (detail - 1.0) * base
        return self.conv(x)