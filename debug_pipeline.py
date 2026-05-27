# ─────────────────────────────────────────────────────────────
# debug_pipeline.py
# Runs ONE image through the full training pipeline and prints
# the state at every stage. Set breakpoints at each # BREAKPOINT
# comment to inspect in VSCode debugger.
# ─────────────────────────────────────────────────────────────

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from mmengine.config import Config
from mmdet.registry import TRANSFORMS
from mmdet.utils import register_all_modules

# ── Register all mmdet + custom modules ───────────────────────
register_all_modules()

# Force custom imports
import modules.raw_preprocessors
import modules.wrappers
import modules.hooks
import datasets.pipelines

# ── Config ────────────────────────────────────────────────────
cfg = Config.fromfile('configs/experiments/exp11_pack3ch_gamma.py')

# ── Pick one image from val set ───────────────────────────────
import json
VAL_JSON = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/data/ROD/yolo/raw/json_raw_coco_mapped/val.json'
DATA_ROOT = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/data/ROD/yolo/'

with open(VAL_JSON) as f:
    coco = json.load(f)

# Get first image and its annotations
img_info  = coco['images'][0]
img_id    = img_info['id']
anns      = [a for a in coco['annotations'] if a['image_id'] == img_id]

print('=' * 60)
print('IMAGE INFO FROM JSON')
print('=' * 60)
print(f"filename:  {img_info['file_name']}")
print(f"JSON w/h:  {img_info['width']} x {img_info['height']}")
print(f"num anns:  {len(anns)}")
for i, ann in enumerate(anns[:3]):
    print(f"  ann[{i}] bbox={ann['bbox']}  cat_id={ann['category_id']}")
# BREAKPOINT 1 — verify JSON bbox coordinates are what you expect

# ── Stage 1: LoadRAWImageFromFile ─────────────────────────────
print('\n' + '=' * 60)
print('STAGE 1: LoadRAWImageFromFile')
print('=' * 60)

loader = TRANSFORMS.build(dict(type='LoadRAWImageFromFile'))
results = {
    'img_path': DATA_ROOT + 'raw/images/val/' + img_info['file_name']
}
results = loader(results)

print(f"img shape:   {results['img'].shape}")
print(f"img dtype:   {results['img'].dtype}")
print(f"img min/max: {results['img'].min():.4f} / {results['img'].max():.4f}")
print(f"img_shape:   {results['img_shape']}")
print(f"ori_shape:   {results['ori_shape']}")
# BREAKPOINT 2 — verify image loads correctly, ori_shape is (1856, 2880)

# ── Stage 2: NormaliseP99 ─────────────────────────────────────
print('\n' + '=' * 60)
print('STAGE 2: NormaliseP99')
print('=' * 60)

normaliser = TRANSFORMS.build(dict(type='NormaliseP99'))
results = normaliser(results)

print(f"img min/max after norm: {results['img'].min():.4f} / {results['img'].max():.4f}")
# BREAKPOINT 3 — verify values are in [0, 1]

# ── Stage 3: PackBayer ────────────────────────────────────────
print('\n' + '=' * 60)
print('STAGE 3: PackBayer (out_channels=3)')
print('=' * 60)

packer = TRANSFORMS.build(dict(type='PackBayer', out_channels=3))
results = packer(results)

print(f"img shape after pack: {results['img'].shape}")
print(f"img_shape after pack: {results['img_shape']}")
print(f"ori_shape after pack: {results['ori_shape']}")
# BREAKPOINT 4 — KEY CHECK
# img shape should be (928, 1440, 3) — half of original
# ori_shape should still be (1856, 2880) — original full res
# QUESTION: do the JSON bbox coords match img_shape or ori_shape?

# ── Stage 4: Check GT bbox vs image size ─────────────────────
print('\n' + '=' * 60)
print('STAGE 4: GT BBOX vs IMAGE SIZE CHECK')
print('=' * 60)

packed_h, packed_w = results['img'].shape[:2]
print(f"Packed image size: {packed_h} x {packed_w}")
print(f"JSON image size:   {img_info['height']} x {img_info['width']}")
print()

for i, ann in enumerate(anns[:3]):
    x, y, w, h = ann['bbox']
    x2, y2 = x + w, y + h
    in_packed = (x2 <= packed_w) and (y2 <= packed_h)
    in_full   = (x2 <= img_info['width']) and (y2 <= img_info['height'])
    print(f"ann[{i}] bbox=[{x:.0f},{y:.0f},{w:.0f},{h:.0f}]")
    print(f"       fits in packed ({packed_w}x{packed_h}): {in_packed}")
    print(f"       fits in full   ({img_info['width']}x{img_info['height']}): {in_full}")
# BREAKPOINT 5 — THIS IS THE KEY DIAGNOSTIC
# If boxes fit in full (2880x1856) but NOT in packed (1440x928):
#   → coordinate mismatch — boxes are 2x too large for the packed image
# If boxes fit in packed:
#   → coordinates are already in packed space — no mismatch

print('\n' + '=' * 60)
print('SUMMARY')
print('=' * 60)
ratio_w = img_info['width'] / packed_w
ratio_h = img_info['height'] / packed_h
print(f"Scale ratio w: {ratio_w:.2f}x  h: {ratio_h:.2f}x")
print(f"Expected ratio for PackBayer: 2.0x")
if abs(ratio_w - 2.0) < 0.01:
    print("⚠ JSON coords are in FULL resolution space")
    print("  GT boxes need to be divided by 2 to match packed image")
else:
    print("✓ JSON coords appear to match packed image space")