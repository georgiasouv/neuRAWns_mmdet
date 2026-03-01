import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmengine.registry import MODELS as MMENGINE_MODELS


@MMENGINE_MODELS.register_module()
@MODELS.register_module()
class Exp005Preprocessor(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        # Shallow feature extractor on 4‑ch Bayer (R, G1, G2, B)
        self.body = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Fuse raw Bayer channels + features
        self.fuse = nn.Conv2d(4 + 32, 16, kernel_size=3, padding=1)

        # Predict per‑pixel gain; clamp to a safe range
        self.gain_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Softplus()   # positive
        )

        # Map fused features to 3‑channel RGB for detectors
        self.rgb_head = nn.Conv2d(16, 3, kernel_size=1)

        self.norm = nn.InstanceNorm2d(3)

    def bayer_to_rggb4(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 1, H, W) Bayer RGGB.
        Returns: (N, 4, H_even/2, W_even/2) = [R, G1, G2, B].
        Crops to even H,W to keep planes aligned.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        assert x.size(1) == 1, f"Expected 1‑channel Bayer, got {x.size(1)}"

        N, _, H, W = x.shape

        # Ensure even sizes
        H_even = H - (H % 2)
        W_even = W - (W % 2)
        x = x[:, :, :H_even, :W_even]      # (N,1,H_even,W_even)

        raw = x[:, 0]                      # (N,H_even,W_even)

        R  = raw[:, 0::2, 0::2]
        G1 = raw[:, 0::2, 1::2]
        G2 = raw[:, 1::2, 0::2]
        B  = raw[:, 1::2, 1::2]

        return torch.stack([R, G1, G2, B], dim=1)  # (N,4,H_even/2,W_even/2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is the padded Bayer batch from DetDataPreprocessor: (N,1,H,W)
        if isinstance(x, list):
            x = torch.stack(x, dim=0)

        N, C, H, W = x.shape

        # Basic HDR‑safe log compression; clamp to avoid NaNs
        x = torch.clamp(x, min=0.0)
        x = torch.log1p(x)

        # Bayer unpack at lower resolution
        rggb4 = self.bayer_to_rggb4(x)          # (N,4,h,w), h=H_even/2, w=W_even/2

        feat = self.body(rggb4)                 # (N,32,h,w)
        fused = torch.cat([rggb4, feat], dim=1) # (N,36,h,w)
        fused = self.fuse(fused)                # (N,16,h,w)

        gain = self.gain_head(fused)            # (N,1,h,w)
        # Clamp gain to a reasonable range to avoid exploding values
        gain = torch.clamp(gain, 0.1, 10.0)
        toned = fused * gain                    # (N,16,h,w)

        rgb = self.rgb_head(toned)              # (N,3,h,w)
        rgb = self.norm(rgb)                    # (N,3,h,w)

        # Final NaN/Inf guard
        rgb = torch.nan_to_num(rgb, nan=0.0, posinf=1e4, neginf=-1e4)

        # Upsample back to original padded size so detectors see (N,3,H,W)
        rgb = F.interpolate(rgb, size=(H, W), mode='bilinear', align_corners=False)

        return rgb
