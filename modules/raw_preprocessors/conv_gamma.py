import torch 
import torch.nn as nn


class ConvGamma(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        if in_channels == 3:

            c = kernel_size // 2             # Set centre pixel of each in→out diagonal to 1
            for i in range(min(in_channels, out_channels)):
                self.conv.weight[i, i, c, c] = 1.0
        elif in_channels == 4:
            c = kernel_size // 2
            self.conv.weight[0, 0, c, c] = 1.0
            self.conv.weight[1, 1, c, c] = 0.5 # G1 + G2 averaged → G  (prior knowledge: they're both green)
            self.conv.weight[1, 2, c, c] = 0.5
            self.conv.weight[2, 3, c, c] = 1.0
            
            
    def forward(self,x):
        x = x**(1/2.2)
        x = self.conv(x)
        return x
    


