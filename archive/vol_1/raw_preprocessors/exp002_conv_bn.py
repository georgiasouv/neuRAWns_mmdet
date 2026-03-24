import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_preprocessor import BasePreprocessor
from mmdet.registry import MODELS


@MODELS.register_module()
class Exp002ConvBN(BasePreprocessor):  # safe, almost fixed, learnable preprocessor
    def __init__(self, norm_threshold=0.95, debug=False):
        super().__init__()
        self.norm_thr = norm_threshold
        self.debug = debug

        # Per‑channel learnable gains (white balance), start at 1
        self.channel_gain = nn.Parameter(torch.ones(3))

        # Very small depthwise conv (3->3), cannot mix colors
        self.conv = nn.Conv2d(
            3, 3, kernel_size=3, stride=1, padding=1, bias=True, groups=3
        )
        self.activation = nn.LeakyReLU(0.1)

        with torch.no_grad():
            # Start as almost identity: zero weights, zero bias
            nn.init.constant_(self.conv.weight, 0.0)
            self.conv.bias.zero_()

    def forward(self, x):
        # x: (B, 1, H, W)
        B, C, H, W = x.shape

        # 1) Pack Bayer RGGB -> 4 channels
        rggb = self.packing(x)          # (B,4,H/2,W/2)

        # 2) Average two greens -> RGB
        R  = rggb[:, 0:1]
        G  = 0.5 * (rggb[:, 1:2] + rggb[:, 2:3])
        Bc = rggb[:, 3:4]
        rgb = torch.cat([R, G, Bc], dim=1)  # (B,3,H/2,W/2)

        # 3) Global luminance normalization (one scalar per image)
        Bn, Cn, Hn, Wn = rgb.shape
        flat = rgb.view(Bn, -1)                     # (B,3*H/2*W/2)
        q = torch.quantile(flat, self.norm_thr, dim=-1, keepdim=True)  # (B,1)
        q = torch.clamp(q, min=1e-6).view(Bn, 1, 1, 1)
        rgb = torch.clamp(rgb / q, 0.0, 1.0)

        # 4) Gamma correction
        rgb = rgb ** (1.0 / 2.2)

        # 5) Per‑channel gain (white balance), constrained
        gain = torch.clamp(self.channel_gain, 0.5, 1.5).view(1, 3, 1, 1)
        rgb = torch.clamp(rgb * gain, 0.0, 1.0)

        # 6) Small depthwise conv + activation (local refinement)
        x = self.conv(rgb)             # (B,3,H/2,W/2)
        x = self.activation(x)
        x = torch.clamp(x, 0.0, 1.0)
        if self.training:
            print(f"[DEBUG] conv bias: {self.conv.bias.data}")
            print(f"[DEBUG] x range after clamp: [{x.min():.4f}, {x.max():.4f}]")

        # 7) Upsample back to original size
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        if self.debug and not self.training:
            ch_mean = x.mean(dim=(0, 2, 3))
            print(f'[Exp002ConvBN] ch_mean={ch_mean}')

        return x

    def packing(self, x):
        """(B,1,H,W) -> (B,4,H/2,W/2) in RGGB order."""
        x = x.squeeze(1)  # (B,H,W)
        R  = x[:, 0::2, 0::2]
        G1 = x[:, 0::2, 1::2]
        G2 = x[:, 1::2, 0::2]
        B  = x[:, 1::2, 1::2]
        return torch.stack([R, G1, G2, B], dim=1)
