from .base_preprocessor import BasePreprocessor
from .helper_functions import identity_conv_init, gamma_warmstart_deltas, apply_lut
from mmdet.registry import MODELS
import torch
import torch.nn as nn


@MODELS.register_module()
class LearnableToneCurve(BasePreprocessor):
    """Per-channel gain → learnable monotonic curve → 1×1 channel mix.
    ~ (C*knots + C + mix) params. Near-zero FLOPS. Thesis-friendly minimum."""
    def __init__(self, in_channels, out_channels, kernel_size=1, knots=16):
        super().__init__()
        self.knots = knots
        self.log_gain = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.curve_deltas = nn.Parameter(gamma_warmstart_deltas(in_channels, knots))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2, bias=True)
        identity_conv_init(self.conv, in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = x * torch.exp(self.log_gain)
        x = apply_lut(x, self.curve_deltas, self.knots)
        return self.conv(x)