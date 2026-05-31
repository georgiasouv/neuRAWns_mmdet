from .base_preprocessor import BasePreprocessor
from .helper_functions import identity_conv_init
from mmdet.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F


@MODELS.register_module()
class LogGammaStack(BasePreprocessor):
    """log-compression → gamma → per-channel WB → channel mix.
    Fully parametric tone curve; fewest params of any nonlinear option."""
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.tensor(2.0))   # softplus → ~2.1
        self.gamma = nn.Parameter(torch.tensor(2.2))
        self.wb = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2, bias=True)
        identity_conv_init(self.conv, in_channels, out_channels, kernel_size)

    def forward(self, x):
        a = F.softplus(self.raw_alpha)
        x = torch.log1p(a * x) / torch.log1p(a)        # adaptive log compression
        x = x.clamp(min=1e-6) ** (1.0 / self.gamma.clamp(min=0.5))
        x = x * self.wb
        return self.conv(x)