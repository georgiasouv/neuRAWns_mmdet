import torch
import torch.nn as nn
from .base_preprocessor import BasePreprocessor
from mmdet.registry import MODELS
    
@MODELS.register_module()
class RAWPreprocess_v1(BasePreprocessor):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self._initialize_weights()
        
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        return x
    
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        nn.init.constant_(self.batchnorm.weight, 1)
        nn.init.constant_(self.batchnorm.bias, 0)