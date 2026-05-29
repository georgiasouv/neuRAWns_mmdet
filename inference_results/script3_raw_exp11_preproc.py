# ─────────────────────────────────────────────────────────────
# script3_raw_exp11_preproc.py  (revised)
#
# RAW Bayer → ConvGamma preprocessor (weights from exp11 checkpoint)
#           → COCO pretrained RTMDet-S (same as script1/script2)
#           → predictions filtered to 5 ROD classes
#
# Design: DECOUPLE preprocessor from detector.
# - ConvGamma weights  → loaded from exp11 checkpoint
# - Detector weights   → original COCO pretrained (untouched)
# This is the correct experimental design: only preprocessing changes,
# the detector is held constant across all three scripts.
# ─────────────────────────────────────────────────────────────

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from mmdet.apis import init_detector, inference_detector

# ── Register custom modules ────────────────────────────────────────────────────
sys.path.insert(0, '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/neuRAWns_mmdet')
import modules.raw_preprocessors
import modules.raw_backbones
import modules.hooks
import datasets.pipelines

# ── Paths ──────────────────────────────────────────────────────────────────────
COCO_CONFIG = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'
COCO_CKPT   = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/.cache/torch/hub/checkpoints/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'
EXP11_CKPT  = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/neuRAWns_mmdet/work_dirs/exp11_pack3ch_gamma/epoch_1.pth'
RAW_PATH    = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/data/ROD/yolo/raw/images/test/day-00116.raw'
OUT_DIR     = 'inference_results/script3_raw_exp11'
SCORE_THR   = 0.3

COCO_ROD_IDS = {0: 'person', 1: 'bicycle', 2: 'car', 6: 'train', 7: 'truck'}

os.makedirs(OUT_DIR, exist_ok=True)


# ── Step 1: Extract ConvGamma weights from exp11 checkpoint ───────────────────
# The checkpoint is a full model state dict. The preprocessor weights live
# under the key prefix 'data_preprocessor.raw_preprocessor.*'.
# We extract only those keys and load them into a standalone ConvGamma —
# completely ignoring all other model weights (backbone, neck, head).

print('Extracting ConvGamma weights from exp11 checkpoint...')
full_state = torch.load(EXP11_CKPT, map_location='cpu', weights_only=False)
state_dict = full_state.get('state_dict', full_state)

prefix = 'data_preprocessor.raw_preprocessor.'
preproc_state = {
    k[len(prefix):]: v
    for k, v in state_dict.items()
    if k.startswith(prefix)
}

print(f'  Found {len(preproc_state)} preprocessor tensors:')
for k, v in preproc_state.items():
    print(f'    {k}: {v.shape}')

# Infer architecture from conv.weight shape: [out_ch, in_ch, kH, kW]
in_channels  = preproc_state['conv.weight'].shape[1]
out_channels = preproc_state['conv.weight'].shape[0]
kernel_size  = preproc_state['conv.weight'].shape[2]
print(f'  Inferred: in={in_channels}  out={out_channels}  kernel={kernel_size}')


# ── Step 2: Build standalone ConvGamma and load weights ───────────────────────
# Defined inline so the script is fully self-contained.
# Must match modules/raw_preprocessors.py exactly.

class ConvGamma(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=True
        )

    def forward(self, x):
        # x: [B, C, H, W], float32, range [0, 1] after p99 norm
        x = x.clamp(min=1e-6) ** (1.0 / 2.2)  # gamma on linear data
        x = self.conv(x)                         # learned spatial mix
        return x

preprocessor = ConvGamma(in_channels, out_channels, kernel_size)
missing, unexpected = preprocessor.load_state_dict(preproc_state, strict=True)
preprocessor.eval().cuda()
print(f'  Loaded — missing keys: {missing}  unexpected: {unexpected}')


# ── Step 3: Load ROD raw image ─────────────────────────────────────────────────
def load_rod_raw(path):
    BIT8, BIT16 = 2**8, 2**16
    raw = np.fromfile(path, dtype=np.uint8).astype(np.float32)
    img = raw[0::3] + raw[1::3] * BIT8 + raw[2::3] * BIT16
    return img.reshape(1856, 2880)

bayer = load_rod_raw(RAW_PATH)
print(f'\n[Step 3] RAW loaded  — shape: {bayer.shape}  range: [{bayer.min():.0f}, {bayer.max():.0f}]')


# ── Step 4: Pack Bayer ─────────────────────────────────────────────────────────
# Automatically uses 3-ch or 4-ch based on what the checkpoint's conv expects.

def pack_bayer_3ch(b):
    R = b[0::2, 0::2]
    G = 0.5 * (b[0::2, 1::2] + b[1::2, 0::2])
    B = b[1::2, 1::2]
    return np.stack([R, G, B], axis=2)   # (928, 1440, 3)

def pack_bayer_4ch(b):
    return np.stack([b[0::2, 0::2], b[0::2, 1::2],
                     b[1::2, 0::2], b[1::2, 1::2]], axis=2)  # (928, 1440, 4)

packed_np = pack_bayer_4ch(bayer) if in_channels == 4 else pack_bayer_3ch(bayer)
print(f'[Step 4] Packed      — shape: {packed_np.shape}')


# ── Step 5: P99 normalisation → [0, 1] ────────────────────────────────────────
# Required BEFORE ConvGamma — gamma is only meaningful on [0, 1] input.

p99 = max(float(np.percentile(packed_np, 99)), 1e-6)
packed_np = np.clip(packed_np / p99, 0.0, 1.0).astype(np.float32)
print(f'[Step 5] P99 norm    — range: [{packed_np.min():.4f}, {packed_np.max():.4f}]')


# ── Step 6: Apply ConvGamma (trained preprocessor) ────────────────────────────
x = torch.from_numpy(packed_np.transpose(2, 0, 1)).unsqueeze(0).cuda()  # [1, C, H, W]

with torch.no_grad():
    x_out = preprocessor(x)   # [1, 3, H, W]

print(f'[Step 6] ConvGamma   — shape: {x_out.shape}  '
      f'range: [{x_out.min():.4f}, {x_out.max():.4f}]')


# ── Step 7: Convert to uint8 BGR for inference_detector ───────────────────────
out_np  = x_out.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # HWC, RGB
out_np  = np.clip(out_np, 0.0, 1.0)
out_bgr = (out_np[:, :, ::-1] * 255).astype(np.uint8)        # HWC, BGR

cv2.imwrite(os.path.join(OUT_DIR, 'day-00116_convgamma_output.png'), out_bgr)
print(f'[Step 7] Saved ConvGamma output image')


# ── Step 8: COCO RTMDet-S inference ───────────────────────────────────────────
print('\nLoading COCO RTMDet-S...')
model  = init_detector(COCO_CONFIG, COCO_CKPT, device='cuda:0')
result = inference_detector(model, out_bgr)

pred   = result.pred_instances
boxes  = pred.bboxes.cpu().numpy()
scores = pred.scores.cpu().numpy()
labels = pred.labels.cpu().numpy()

print(f'Raw predictions: {len(boxes)}')
if len(boxes) > 0:
    print(f'Score range: [{scores.min():.4f}, {scores.max():.4f}]')


# ── Step 9: Filter to ROD classes + NMS ───────────────────────────────────────
from torchvision.ops import nms

keep  = (scores > SCORE_THR) & np.isin(labels, list(COCO_ROD_IDS.keys()))
boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

if len(boxes) > 0:
    idx    = nms(torch.tensor(boxes, dtype=torch.float32),
                 torch.tensor(scores, dtype=torch.float32),
                 iou_threshold=0.45).numpy()
    boxes, scores, labels = boxes[idx], scores[idx], labels[idx]

print(f'Predictions after filter+NMS: {len(boxes)}')


# ── Step 10: Save .txt ────────────────────────────────────────────────────────
txt_path = os.path.join(OUT_DIR, 'day-00116_preds.txt')
with open(txt_path, 'w') as f:
    f.write('Script 3 — RAW → ConvGamma (exp11 weights) → COCO RTMDet-S\n')
    f.write(f'RAW input:       {RAW_PATH}\n')
    f.write(f'Preprocessor:    ConvGamma in={in_channels} out={out_channels} k={kernel_size}\n')
    f.write(f'Preproc weights: {EXP11_CKPT}\n')
    f.write(f'Detector:        COCO RTMDet-S (original weights, unchanged)\n')
    f.write(f'Score thr:       {SCORE_THR}\n')
    f.write(f'Total predictions: {len(boxes)}\n')
    f.write('─' * 60 + '\n')
    f.write(f'{"#":<5} {"class":<12} {"score":<8} {"x1":>6} {"y1":>6} {"x2":>6} {"y2":>6}\n')
    f.write('─' * 60 + '\n')
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        cls_name = COCO_ROD_IDS.get(int(label), f'cls_{label}')
        x1, y1, x2, y2 = box.astype(int)
        f.write(f'{i:<5} {cls_name:<12} {score:.4f}   {x1:>6} {y1:>6} {x2:>6} {y2:>6}\n')
print(f'Saved {txt_path}')


# ── Step 11: Save annotated image ─────────────────────────────────────────────
COLOURS = {'person': (0,255,0), 'bicycle': (255,165,0), 'car': (0,0,255),
           'train': (255,0,255), 'truck': (0,255,255)}

vis = out_bgr.copy()
for box, score, label in zip(boxes, scores, labels):
    x1, y1, x2, y2 = box.astype(int)
    cls_name = COCO_ROD_IDS.get(int(label), f'cls_{label}')
    c = COLOURS.get(cls_name, (255,255,255))
    cv2.rectangle(vis, (x1,y1), (x2,y2), c, 2)
    cv2.putText(vis, f'{cls_name} {score:.2f}', (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

png_path = os.path.join(OUT_DIR, 'day-00116_preds.png')
cv2.imwrite(png_path, vis)
print(f'Saved {png_path}')