from .base_preprocessor import BasePreprocessor
from .helper_functions import identity_conv_init
from mmdet.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F


@MODELS.register_module()
class ResidualRefiner(BasePreprocessor):
    """LogGamma-style tone backbone + small depthwise-separable residual.
    Residual is scaled by a learnable factor init near 0 → starts as backbone."""
    def __init__(self, in_channels, out_channels, kernel_size=3, hidden=16):
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.tensor(2.0))
        self.gamma = nn.Parameter(torch.tensor(2.2))

        self.dw = nn.Conv2d(in_channels, in_channels, 3, padding=1,
                            groups=in_channels)
        self.pw1 = nn.Conv2d(in_channels, hidden, 1)
        self.pw2 = nn.Conv2d(hidden, in_channels, 1)
        self.act = nn.GELU()
        self.res_scale = nn.Parameter(torch.zeros(1))   # passthrough at init

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2, bias=True)
        identity_conv_init(self.conv, in_channels, out_channels, kernel_size)

    def forward(self, x):
        a = F.softplus(self.raw_alpha)
        base = torch.log1p(a * x) / torch.log1p(a)
        base = base.clamp(min=1e-6) ** (1.0 / self.gamma.clamp(min=0.5))
        r = self.pw2(self.act(self.pw1(self.act(self.dw(base)))))
        x = base + self.res_scale * r
        return self.conv(x)