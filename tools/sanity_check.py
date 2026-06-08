#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────
# tools/sanity_check.py
# Pre-flight for the ensemble config. Builds the model, runs ONE
# train batch + one predict, and asserts every invariant the smoke
# test would eventually reveal — but in seconds, with named checks.
#
#   python tools/sanity_check.py configs/experiments/exp30_ensemble_curve_lin.py
#
# Exits 0 only if all checks pass.
# ─────────────────────────────────────────────────────────────
import argparse
import copy
import sys

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

PASS, FAIL = '  [PASS]', '  [FAIL]'
results = []


def check(name, cond, detail=''):
    ok = bool(cond)
    results.append(ok)
    print(f'{PASS if ok else FAIL} {name}' + (f'  — {detail}' if detail else ''))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    args = ap.parse_args()

    register_all_modules()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}\n')

    # ── 1. config resolves ────────────────────────────────────
    cfg = Config.fromfile(args.config)
    det_types = [d['type'] for d in cfg.model['detector_cfgs']]
    check('config resolves; model + loops present',
          cfg.model['type'] == 'MultiDetectorModel'
          and cfg.val_cfg['type'] == 'EnsembleValLoop',
          f"{cfg.model['type']} / {cfg.val_cfg['type']}")
    check('three detector cfgs interpolated',
          len(det_types) == 3 and all(isinstance(t, str) for t in det_types),
          str(det_types))

    # ── 2. model builds + checkpoints load ────────────────────
    model = MODELS.build(cfg.model).to(device)
    model.eval()
    check('model built, detector_names set',
          hasattr(model, 'detector_names') and len(model.detector_names) == 3,
          str(getattr(model, 'detector_names', None)))

    # ── 3. freeze: only preprocessor trains ───────────────────
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    pre_trainable = [n for n in trainable if n.startswith('preprocessor.')]
    det_trainable = [n for n in trainable if n.startswith('detectors.')]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check('NO detector parameter is trainable',
          len(det_trainable) == 0,
          f'{len(det_trainable)} detector params want grad')
    check('preprocessor IS trainable',
          len(pre_trainable) > 0, f'{n_train/1e3:.1f}k trainable params')

    # ── 4. detectors pinned to eval AFTER model.train() ───────
    model.train()                          # mmengine does this before the loop
    det_modes = [m.training for d in model.detectors
                 for m in d.modules() if isinstance(m, nn.BatchNorm2d)]
    check('all detector BatchNorm in eval() after model.train()',
          len(det_modes) == 0 or not any(det_modes),
          f'{sum(det_modes)}/{len(det_modes)} BN still in train mode')

    # ── 5. one real train batch ───────────────────────────────
    loader = Runner.build_dataloader(cfg.train_dataloader)
    batch = next(iter(loader))
    data = model.data_preprocessor(copy.deepcopy(batch), training=True)
    losses = model.loss(data['inputs'], data['data_samples'])

    # 5a. shared enhancement contract
    ni, en = model._last_norm_input, model._last_enhanced
    check('normalised input is 4-channel', ni.shape[1] == 4, str(tuple(ni.shape)))
    check('enhanced is 3-channel', en.shape[1] == 3, str(tuple(en.shape)))
    check('enhanced is on a [0,255]-ish scale (Option A)',
          en.max().item() > 5.0 and en.max().item() < 400.0,
          f'enhanced max={en.max().item():.1f} mean={en.mean().item():.1f}')

    # 5b. identity init (LocalCurve starts ~identity)
    ratio = en.float().mean().item() / (255 * ni.float().mean().item() + 1e-12)
    check('module starts near identity (ratio O(1))',
          0.2 <= ratio <= 5.0, f'enhanced.mean/(255*input.mean)={ratio:.3f}')

    # 5c. per-detector input contract (post each detector's own norm)
    for name, s in model._last_det_input_stats.items():
        check(f'{name} backbone input in-band (post-COCO-norm)',
              -2.5 <= s['mean'] <= 2.5 and 0.2 <= s['std'] <= 4.0,
              f"mean={s['mean']:.2f} std={s['std']:.2f}")

    # 5d. losses: namespaced per detector, all finite
    by_det = {n: [k for k in losses if k.startswith(n + '.')]
              for n in model.detector_names}
    check('losses namespaced for every detector',
          all(by_det[n] for n in model.detector_names),
          {n: len(v) for n, v in by_det.items()})
    finite = all(torch.isfinite(v).all() for v in losses.values()
                 if torch.is_tensor(v))
    check('all loss values finite', finite)

    # ── 6. gradient flows to preprocessor ONLY ────────────────
    total = sum(v for v in losses.values() if torch.is_tensor(v) and v.ndim == 0)
    model.zero_grad(set_to_none=True)
    total.backward()
    pre_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.preprocessor.parameters())
    det_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for d in model.detectors for p in d.parameters())
    check('preprocessor RECEIVES gradient', pre_grad)
    check('detectors receive NO gradient', not det_grad)

    # ── 7. BN running stats did NOT move (the freeze proof) ───
    model.eval()
    bn = next((m for d in model.detectors for m in d.modules()
               if isinstance(m, nn.BatchNorm2d)
               and m.track_running_stats), None)
    if bn is not None:
        before = bn.running_mean.clone()
        model.train()
        d2 = model.data_preprocessor(copy.deepcopy(batch), training=True)
        model.loss(d2['inputs'], d2['data_samples'])
        moved = not torch.allclose(before, bn.running_mean)
        check('detector BN running_mean UNCHANGED after fwd in train mode',
              not moved, 'stats moved — train() override not working' if moved
              else 'frozen')
    else:
        print('  [skip] no trackable BN found (all FrozenBN) — fine')

    # ── 8. predict path per detector ──────────────────────────
    model.eval()
    with torch.no_grad():
        for name in model.detector_names:
            model.eval_detector = name
            d3 = model.data_preprocessor(copy.deepcopy(batch), training=False)
            out = model.predict(d3['inputs'], d3['data_samples'])
            check(f'predict() returns instances via {name}',
                  len(out) > 0 and hasattr(out[0], 'pred_instances'))

    print(f'\n{sum(results)}/{len(results)} checks passed')
    sys.exit(0 if all(results) else 1)


if __name__ == '__main__':
    main()