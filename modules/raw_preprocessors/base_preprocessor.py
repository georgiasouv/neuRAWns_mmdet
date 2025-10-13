import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BasePreprocessor(nn.Module, ABC):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config_dict(self):
        return {
            'type': self.__class__.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'num_parameters': self.count_parameters()
        }
    
     
    

