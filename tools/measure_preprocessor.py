"""
Measure parameters, GFLOPs, and latency of the preprocessor module only.

Usage:
    python tools/measure_preprocessor.py experiments/exp11_pack3ch_gamma.py
"""

import torch
import time
import argparse
from mmengine.config import Config
from mmdet.registry import MODELS

# ── thop for FLOPs ────────────────────────────────────────────
# pip install thop
from thop import profile, clever_format


def build_preprocessor(cfg):
    """Pull the preprocessor_cfg out of the model config and build it."""
    preprocessor_cfg = cfg.model.data_preprocessor.preprocessor_cfg
    return MODELS.build(preprocessor_cfg)


def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def measure_flops(module, dummy_input):
    """Uses thop to count MACs (multiply-accumulate ops). 1 MAC ≈ 2 FLOPs."""
    macs, params = profile(module, inputs=(dummy_input,), verbose=False)
    macs_str, params_str = clever_format([macs, params], "%.3f")
    return macs, macs_str


def measure_latency(module, dummy_input, n_warmup=50, n_runs=200):
    """
    GPU latency using CUDA events — more accurate than time.time() because
    CUDA is async. We warm up first to avoid measuring JIT / memory alloc.
    """
    module = module.cuda().eval()
    dummy_input = dummy_input.cuda()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = module(dummy_input)

    # Timed runs
    start_event = torch.cuda.Event(enable_timing=True)
    end_event   = torch.cuda.Event(enable_timing=True)
    times = []

    with torch.no_grad():
        for _ in range(n_runs):
            start_event.record()
            _ = module(dummy_input)
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))  # milliseconds

    mean_ms = sum(times) / len(times)
    std_ms  = (sum((t - mean_ms)**2 for t in times) / len(times)) ** 0.5
    return mean_ms, std_ms


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to experiment config')
    parser.add_argument('--in-channels', type=int, default=3)
    # Use a representative input size (after PackBayer, half resolution)
    parser.add_argument('--H', type=int, default=400)   # ~800/2 for packed
    parser.add_argument('--W', type=int, default=667)   # ~1333/2 for packed
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    # Register your custom modules first
    import mmdet.datasets  # noqa
    import mmdet.models    # noqa
    # Add your custom module imports here e.g.:
    # from neurawns.models.preprocessors import ConvGamma  # noqa

    preprocessor = build_preprocessor(cfg)
    preprocessor.eval()

    # Dummy input: [B=1, C, H, W], values in [0,1] (post-NormaliseP99)
    dummy = torch.rand(1, args.in_channels, args.H, args.W)

    # ── Parameters ────────────────────────────────────────────
    total_params, trainable_params = count_parameters(preprocessor)
    print(f"\n{'='*50}")
    print(f"Config: {args.config}")
    print(f"{'='*50}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ── FLOPs ─────────────────────────────────────────────────
    macs, macs_str = measure_flops(preprocessor, dummy)
    print(f"MACs (≈GFLOPs/2):     {macs_str}  ({macs/1e9:.4f} GMACs)")

    # ── Latency ───────────────────────────────────────────────
    if torch.cuda.is_available():
        mean_ms, std_ms = measure_latency(preprocessor, dummy)
        print(f"GPU Latency:          {mean_ms:.3f} ± {std_ms:.3f} ms")
        print(f"  → Implies max FPS:  {1000/mean_ms:.1f} FPS (preprocessor only)")
    else:
        print("No CUDA — skipping latency measurement")

    print(f"{'='*50}\n")
    
    """
        ** count_parameters — counts all vs trainable. In exp11 all conv params are trainable, but once you add learnable gamma in exp16/17, this distinction matters
        ** thop.profile — traces through the module and counts multiply-accumulate ops. Reports MACs not FLOPs; FLOPs ≈ 2× MACs
        ** CUDA events — start_event.record() → end_event.record() → torch.cuda.synchronize() is the correct pattern because CUDA kernels are asynchronous; time.time() would give you wrong numbers
        ** Warmup runs — the first N passes allocate memory and JIT-compile kernels; excluding them gives you steady-state latency
    """