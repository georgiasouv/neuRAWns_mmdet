import torch
import torch.nn as nn

class ConvGammaGain(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.01) # no Kaiming initialisation which is use for RELU
        nn.init.zeros_(self.conv.bias)
        self.gain = nn.Parameter(torch.ones(out_channels, 1, 1))    # For a per-channel gain,  one scalar per output channel, shaped for broadcasting over [B, C, H, W]
                                                                    #  (1, 1) lets PyTorch broadcast the gain across H and W automatically.
        
        
    def forward(self,x):
        x = x**(1/2.2)
        x = self.conv(x)
        x = x * self.gain
        return x
            