#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────
# tools/parity_check.py
# Verifies the deployment path reproduces the training path.
#
# Builds BOTH models:
#   A) MultiDetectorWrapper from the training config + trained ckpt,
#      with eval_detector = --detector
#   B) SingleDetectorDeploy from the deploy config (which loads the
#      same trained preprocessor + the same detector weights)
# feeds them IDENTICAL val batches through their own test_step
# (so each runs its own top-level data_preprocessor), and compares
# predictions box-for-box.
#
# If parity fails, the deploy config diverges from training —
# usually the data_preprocessor block (normalisation constants) or
# the preprocessor cfg (variant / knots / out_scale).
#
# Usage:
#   python tools/parity_check.py \
#       --train-config configs/parallel/exp_p01.py \
#       --deploy-config configs/deploy/deploy_retinanet.py \
#       --checkpoint work_dirs/exp_p01/epoch_50.pth \
#       --detector retinanet \
#       --num-batches 3 [--benchmark]
# ─────────────────────────────────────────────────────────────
import argparse
import time

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train-config', required=True)
    p.add_argument('--deploy-config', required=True)
    p.add_argument('--checkpoint', required=True,
                   help='trained MultiDetectorWrapper checkpoint')
    p.add_argument('--detector', required=True,
                   help='detector name as keyed in the wrapper')
    p.add_argument('--num-batches', type=int, default=3)
    p.add_argument('--atol-bbox', type=float, default=1e-3)
    p.add_argument('--atol-score', type=float, default=1e-4)
    p.add_argument('--benchmark', action='store_true',
                   help='also time the tone module alone (100 iters)')
    return p.parse_args()


def compare_batch(preds_a, preds_b, atol_bbox, atol_score):
    """Compare two lists of DetDataSample. Returns (ok, report)."""
    msgs, ok = [], True
    for i, (sa, sb) in enumerate(zip(preds_a, preds_b)):
        pa, pb = sa.pred_instances, sb.pred_instances
        if len(pa) != len(pb):
            ok = False
            msgs.append(f'  img{i}: count {len(pa)} vs {len(pb)} MISMATCH')
            continue
        if len(pa) == 0:
            msgs.append(f'  img{i}: 0 detections in both — OK')
            continue
        d_box = (pa.bboxes - pb.bboxes).abs().max().item()
        d_scr = (pa.scores - pb.scores).abs().max().item()
        lbl_ok = bool((pa.labels == pb.labels).all())
        this_ok = d_box <= atol_bbox and d_scr <= atol_score and lbl_ok
        ok &= this_ok
        msgs.append(f'  img{i}: n={len(pa)} max|dbox|={d_box:.2e} '
                    f'max|dscore|={d_scr:.2e} labels_equal={lbl_ok} '
                    f'-> {"OK" if this_ok else "MISMATCH"}')
    return ok, msgs


def main():
    args = parse_args()
    register_all_modules()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_cfg = Config.fromfile(args.train_config)
    deploy_cfg = Config.fromfile(args.deploy_config)

    # A) training-path model
    wrapper = MODELS.build(train_cfg.model)
    load_checkpoint(wrapper, args.checkpoint, map_location='cpu')
    assert args.detector in wrapper.detector_names, (
        f'{args.detector!r} not in {wrapper.detector_names}')
    wrapper.eval_detector = args.detector
    wrapper.to(device).eval()

    # B) deployment-path model (its config loads its own checkpoints)
    deploy = MODELS.build(deploy_cfg.model)
    deploy.to(device).eval()

    # Identical data: the training config's val pipeline.
    loader = Runner.build_dataloader(train_cfg.val_dataloader)

    all_ok = True
    with torch.no_grad():
        for bi, data_batch in enumerate(loader):
            if bi >= args.num_batches:
                break
            preds_a = wrapper.test_step(data_batch)
            preds_b = deploy.test_step(data_batch)
            ok, msgs = compare_batch(preds_a, preds_b,
                                     args.atol_bbox, args.atol_score)
            all_ok &= ok
            print(f'batch {bi}: {"PASS" if ok else "FAIL"}')
            for m in msgs:
                print(m)

    print('\n' + ('PARITY: PASS — deployment path reproduces the '
                  'training path.' if all_ok else
                  'PARITY: FAIL — deploy config diverges from training. '
                  'Diff the data_preprocessor and preprocessor blocks '
                  'of the two configs first.'))

    if args.benchmark:
        x = None
        for data_batch in loader:
            d = deploy.data_preprocessor(data_batch, False)
            x = d['inputs'] if isinstance(d, dict) else d[0]
            break
        with torch.no_grad():
            for _ in range(10):                      # warmup
                deploy.preprocessor(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                deploy.preprocessor(x)
            if device == 'cuda':
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 100 * 1000
        n_par = sum(p.numel() for p in deploy.preprocessor.parameters())
        print(f'\nTone module: {n_par/1e3:.1f}k params, '
              f'{dt:.2f} ms/batch (input {tuple(x.shape)}, {device})')

    raise SystemExit(0 if all_ok else 1)


if __name__ == '__main__':
    main()