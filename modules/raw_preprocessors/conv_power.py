import torch 
import torch.nn as nn
from .base_preprocessor import BasePreprocessor
from mmdet.registry import MODELS
import math


@MODELS.register_module()
class ConvPower(BasePreprocessor):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=True)
        self.log_pow = nn.Parameter(
            torch.full((1, in_channels, 1, 1), math.log(2.2)))
        with torch.no_grad():
            nn.init.zeros_(self.conv.weight)
            assert self.conv.bias is not None
            nn.init.zeros_(self.conv.bias)
            c = kernel_size // 2
            if in_channels == 3:
                for i in range(min(in_channels, out_channels)):
                    self.conv.weight[i, i, c, c] = 1.0
            elif in_channels == 4:
                self.conv.weight[0, 0, c, c] = 1.0
                self.conv.weight[1, 1, c, c] = 0.5 # G1 + G2 averaged → G  (prior knowledge: they're both green)
                self.conv.weight[1, 2, c, c] = 0.5
                self.conv.weight[2, 3, c, c] = 1.0
                
            
    def forward(self,x):
        # exponent = torch.exp(self.log_pow) # to ensure pow remains positive
        exponent = torch.exp(self.log_pow.clamp(-2, 2))  # constrains gamma to [e^-2, e^2] ≈ [0.13, 7.4]
        x = x.clamp(min=1e-6) ** (1 / exponent)
        x = self.conv(x)
        return x
    


# no need for activation(RELU) || BtachNorm as I have only one convolution