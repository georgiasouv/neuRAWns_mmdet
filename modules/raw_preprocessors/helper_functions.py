import torch
import torch.nn as nn
import torch.nn.functional as F


def identity_conv_init(conv, in_channels, out_channels, kernel_size):
    """Warm-start a conv as a near-passthrough (your ConvGammaGain convention)."""
    with torch.no_grad():
        nn.init.zeros_(conv.weight)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)
        c = kernel_size // 2
        if in_channels == 3:
            for i in range(min(in_channels, out_channels)):
                conv.weight[i, i, c, c] = 1.0
        elif in_channels == 4:
            conv.weight[0, 0, c, c] = 1.0
            conv.weight[1, 1, c, c] = 0.5   # G1 + G2 → G
            conv.weight[1, 2, c, c] = 0.5
            conv.weight[2, 3, c, c] = 1.0


def gamma_warmstart_deltas(channels, knots, gamma=2.2):
    """Pre-softplus deltas whose cumsum(softplus(.)) ≈ x**(1/gamma) curve."""
    knot_x = torch.linspace(0, 1, knots)
    curve = knot_x.clamp(min=1e-6) ** (1.0 / gamma)
    deltas = torch.diff(curve, prepend=torch.zeros(1)).clamp(min=1e-4)
    raw = torch.log(torch.expm1(deltas))          # inverse softplus
    return raw.unsqueeze(0).repeat(channels, 1)   # (C, K)


def apply_lut(x, curve_deltas, knots):
    """Per-channel monotonic 1D LUT via linear interpolation. x in [0,1]."""
    curve = torch.cumsum(F.softplus(curve_deltas), dim=1)   # (C, K)
    curve = curve / curve[:, -1:].clamp_min(1e-6)           # normalise to [0,1]
    B, C, H, W = x.shape
    pos = x.clamp(0, 1) * (knots - 1)
    lo = pos.floor().long().clamp(0, knots - 2)
    frac = pos - lo.float()
    curve_b = curve.view(1, C, knots, 1, 1).expand(B, C, knots, H, W)
    g_lo = torch.gather(curve_b, 2, lo.unsqueeze(2)).squeeze(2)
    g_hi = torch.gather(curve_b, 2, (lo + 1).unsqueeze(2)).squeeze(2)
    return g_lo * (1 - frac) + g_hi * frac