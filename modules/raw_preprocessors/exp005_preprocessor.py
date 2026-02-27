import torch.nn as nn
import torch
from mmdet.registry import MODELS

@MODELS.register_module()
class Exp005Preprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),    #================================================
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.InstanceNorm2d(3),
        )
    
    def forward(self, x):
        return x + self.net(x)  # residual correction with normalized output