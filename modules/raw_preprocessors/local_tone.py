import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_preprocessor import BasePreprocessor
from mmdet.registry import MODELS


# ─────────────────────────────────────────────────────────────────────────────
# Spatially-adaptive tone-mapping preprocessors.  Three variants share one
# backbone so the params-vs-mAP comparison is fair (only the spatial mechanism
# differs).  Operation order matches the Conv* family and real ISPs:
#
#       tone-map (in raw domain)  ->  channel mix (4ch -> 3ch)
#
# This keeps the channel mix operating on dynamic-range-compressed values
# instead of raw 20-stop HDR, which is far better conditioned (a mix-first
# order destroys near-zero night signal before the curve can lift it).
#
# Cost model (what makes this real-time):
#   - predictor CNN runs ONLY on a fixed thumb_size thumbnail -> O(1) in H,W
#   - everything at full res is pointwise (tone apply + 1x1 mix) -> cheap
#
# Input  : [B, in_channels, H, W] in [0,1]  (NormaliseP99 / fixed-scale upstream)
# Output : [B, out_channels, H, W]
# ─────────────────────────────────────────────────────────────────────────────
class _LocalToneBase(BasePreprocessor):
    # P = number of predicted parameters per (input-channel, grid-cell).
    params_per_cell = None

    def __init__(self,
                 in_channels=4,
                 out_channels=3,
                 grid_h=8,
                 grid_w=8,
                 thumb_size=64,
                 hidden=32,
                 knot_spacing='log',     # 'log' | 'uniform'
                 knot_eps=1e-5,
                 out_scale=1.0):         # scale output before detector normalisation
        super().__init__(in_channels, out_channels)
        assert self.params_per_cell is not None, "subclass must set params_per_cell"
        assert knot_spacing in ('log', 'uniform')
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.thumb_size = thumb_size
        self.knot_spacing = knot_spacing
        self.knot_eps = knot_eps
        self.out_scale = out_scale

        # predictor CNN on the thumbnail. Sees raw in_channels (before mixing),
        # so it can reason about per-channel exposure. Cost is constant in H,W.
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        # head predicts a grid of tone params, PER INPUT CHANNEL
        out_dim = in_channels * grid_h * grid_w * self.params_per_cell
        self.head = nn.Linear(hidden, out_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)              # identity tone at init

        # final channel mix, applied AFTER tone-mapping. Bias zeroed so the
        # mix starts neutral and doesn't inject offsets into compressed signal.
        self.mix = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.mix.bias)

    # --- shared helpers --------------------------------------------------------
    def _predict_grid(self, x):
        """x:[B,Cin,H,W] -> grid params [B,Cin,gh,gw,P] from the thumbnail."""
        thumb = F.interpolate(x, size=(self.thumb_size, self.thumb_size),
                              mode='bilinear', align_corners=False)
        feat = self.predictor(thumb).flatten(1)
        raw = self.head(feat)
        return raw.reshape(-1, self.in_channels,
                           self.grid_h, self.grid_w, self.params_per_cell)

    @staticmethod
    def _upsample_grid(grid_bchwp, H, W):
        """[B,C,gh,gw,P] -> [B,C,H,W,P] via smooth bilinear spatial upsampling."""
        B, C, gh, gw, P = grid_bchwp.shape
        g = grid_bchwp.permute(0, 1, 4, 2, 3).reshape(B, C * P, gh, gw)
        g = F.interpolate(g, size=(H, W), mode='bilinear', align_corners=False)
        return g.reshape(B, C, P, H, W).permute(0, 1, 3, 4, 2)

    def _knot_positions(self, x, K):
        """Map x in [0,1] to knot-axis position [0,K-1].
        'log' spreads knots in log-space so ~half sit below the day/night
        boundary where hard objects live (essential for fixed-linear input)."""
        if self.knot_spacing == 'uniform':
            return x.clamp(0, 1) * (K - 1)
        lx = torch.log(x.clamp(min=self.knot_eps))
        lo = torch.log(torch.tensor(self.knot_eps, device=x.device, dtype=x.dtype))
        return (lx - lo) / (0.0 - lo) * (K - 1)

    def forward(self, x):
        toned = self._tone(x)                    # [B,Cin,H,W] tone-mapped, per channel
        out = self.mix(toned)                    # [B,Cout,H,W] channel mix
        return out * self.out_scale

    def _tone(self, x):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Variant A — spatial AFFINE grid (HDRNet-style, lightest moving part).
# Per (input-channel, cell): gain & bias.  tone = x*gain + bias.
# gain = exp(log_gain) > 0.  Identity at init (head=0 -> gain=1, bias=0).
# ─────────────────────────────────────────────────────────────────────────────
@MODELS.register_module()
class LocalAffine(_LocalToneBase):
    params_per_cell = 2   # [log_gain, bias]

    def _tone(self, x):
        B, C, H, W = x.shape
        grid = self._predict_grid(x)
        up = self._upsample_grid(grid, H, W)             # [B,C,H,W,2]
        gain = torch.exp(up[..., 0])
        bias = up[..., 1]
        return (x * gain + bias).clamp(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Variant B — spatial CURVE grid (most expressive).
# Per (input-channel, cell): K knot-deltas -> monotonic curve in [0,1].
# ─────────────────────────────────────────────────────────────────────────────
@MODELS.register_module()
class LocalCurve(_LocalToneBase):
    def __init__(self, *args, n_knots=16, **kwargs):
        self.n_knots = n_knots
        self.params_per_cell = n_knots
        super().__init__(*args, **kwargs)

    @staticmethod
    def _build_curves(deltas):
        inc = F.softplus(deltas)
        curve = torch.cumsum(inc, dim=-1)
        curve = curve - curve[..., :1]
        curve = curve / (curve[..., -1:] + 1e-6)
        return curve

    def _tone(self, x):
        B, C, H, W = x.shape
        grid = self._predict_grid(x)
        up = self._upsample_grid(grid, H, W)             # [B,C,H,W,K] deltas
        curve = self._build_curves(up)                   # monotonic
        K = curve.shape[-1]
        pos = self._knot_positions(x, K)
        lo = pos.floor().long().clamp(0, K - 2)
        hi = lo + 1
        frac = pos - lo.float()
        y_lo = torch.gather(curve, -1, lo.unsqueeze(-1)).squeeze(-1)
        y_hi = torch.gather(curve, -1, hi.unsqueeze(-1)).squeeze(-1)
        return y_lo + frac * (y_hi - y_lo)


# ─────────────────────────────────────────────────────────────────────────────
# Variant C — HYBRID (global per-channel curve + spatial gain, middle ground).
# ─────────────────────────────────────────────────────────────────────────────
@MODELS.register_module()
class LocalHybrid(_LocalToneBase):
    params_per_cell = 1   # [log_gain] per cell

    def __init__(self, *args, n_knots=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_knots = n_knots
        self.global_deltas = nn.Parameter(
            torch.zeros(1, self.in_channels, n_knots))   # linear curve at init

    def _tone(self, x):
        B, C, H, W = x.shape

        # global per-channel monotonic curve
        inc = F.softplus(self.global_deltas)
        curve = torch.cumsum(inc, dim=-1)
        curve = curve - curve[..., :1]
        curve = curve / (curve[..., -1:] + 1e-6)         # [1,C,K]
        K = curve.shape[-1]
        curve = curve.expand(B, C, K)
        pos = self._knot_positions(x, K)
        lo = pos.floor().long().clamp(0, K - 2)
        hi = lo + 1
        frac = pos - lo.float()
        cexp = curve.unsqueeze(-1).expand(B, C, K, H * W)
        y_lo = torch.gather(cexp, 2, lo.reshape(B, C, 1, H * W)).reshape(B, C, H, W)
        y_hi = torch.gather(cexp, 2, hi.reshape(B, C, 1, H * W)).reshape(B, C, H, W)
        toned = y_lo + frac * (y_hi - y_lo)

        # spatial gain map
        grid = self._predict_grid(x)
        up = self._upsample_grid(grid, H, W)             # [B,C,H,W,1]
        gain = torch.exp(up[..., 0])
        return (toned * gain).clamp(0, 1)