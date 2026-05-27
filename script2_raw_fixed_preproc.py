# ─────────────────────────────────────────────────────────────
# script2_raw_fixed_preproc.py
# RAW Bayer image → fixed preprocessing (pack, p99, AWB, gamma)
#                 → COCO pretrained RTMDet-S → predictions
#
# Same detector as script1 — only the input changes.
# Lets us isolate: does manual fixed preprocessing produce
# sensible detections, independent of any learned module?
# ─────────────────────────────────────────────────────────────

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from mmdet.apis import init_detector, inference_detector

# ── Paths ──────────────────────────────────────────────────────────────────────
CONFIG     = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'
CHECKPOINT = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/.cache/torch/hub/checkpoints/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'

# ROD raw file corresponding to day-00116 — update extension if needed
RAW_PATH   = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/data/ROD/yolo/raw/images/test/day-00116.raw'
OUT_DIR    = 'inference_results/script2_raw_fixed'

SCORE_THR  = 0.3

# COCO IDs for ROD-relevant classes (same as script1)
COCO_ROD_IDS = {0: 'person', 1: 'bicycle', 2: 'car', 6: 'train', 7: 'truck'}

os.makedirs(OUT_DIR, exist_ok=True)


# ── Step 1: Load ROD raw image ─────────────────────────────────────────────────
# ROD is a 24-bit HDR sensor (IMX490).
# Each pixel is stored as 3 consecutive uint8 bytes encoding a 24-bit value.
# Byte layout: value = byte0 + byte1*256 + byte2*65536
# Output shape after reshape: (1856, 2880), dtype float32

def load_rod_raw(path: str) -> np.ndarray:
    BIT8,  BIT16 = 2**8, 2**16
    raw = np.fromfile(path, dtype=np.uint8).astype(np.float32)
    img = raw[0::3] + raw[1::3] * BIT8 + raw[2::3] * BIT16   # 24-bit reconstruction
    return img.reshape(1856, 2880)                              # (H, W), float32, range [0, ~16M]

bayer = load_rod_raw(RAW_PATH)
print(f'[Step 1] Loaded RAW  — shape: {bayer.shape}  range: [{bayer.min():.0f}, {bayer.max():.0f}]')


# ── Step 2: Pack Bayer → 3-channel (demosaic) ─────────────────────────────────
# ROD uses RGGB layout:
#   row even, col even  → R
#   row even, col odd   → G1
#   row odd,  col even  → G2
#   row odd,  col odd   → B
#
# We average G1+G2 into a single G channel.
# Output: (928, 1440, 3), float32, same value range as input.

def pack_bayer_3ch(bayer: np.ndarray) -> np.ndarray:
    R  = bayer[0::2, 0::2]
    G1 = bayer[0::2, 1::2]
    G2 = bayer[1::2, 0::2]
    B  = bayer[1::2, 1::2]
    G  = 0.5 * (G1 + G2)
    return np.stack([R, G, B], axis=2)   # (H/2, W/2, 3)

packed = pack_bayer_3ch(bayer)
print(f'[Step 2] Packed      — shape: {packed.shape}  range: [{packed.min():.0f}, {packed.max():.0f}]')


# ── Step 3: P99 normalisation → [0, 1] ────────────────────────────────────────
# Done BEFORE AWB so that gains are computed on well-scaled data.
# We compute a single global p99 (not per-channel) to preserve relative
# channel ratios — those ratios carry colour information that AWB will use.

def normalise_p99(img: np.ndarray) -> np.ndarray:
    p99 = np.percentile(img, 99)
    p99 = max(p99, 1e-6)
    return np.clip(img / p99, 0.0, 1.0)

packed = normalise_p99(packed)
print(f'[Step 3] P99 norm    — range: [{packed.min():.4f}, {packed.max():.4f}]')


# ── Step 4: Gray-world AWB ────────────────────────────────────────────────────
# Applied on linear (pre-gamma) data — this is the correct point in the ISP.
# Gray world assumption: the scene average should be achromatic (R≈G≈B).
# We scale each channel so that its spatial mean equals the grand mean.

def gray_world_awb(img: np.ndarray) -> np.ndarray:
    # img: (H, W, 3) in [0, 1], RGB order
    means       = img.mean(axis=(0, 1))          # [R_mean, G_mean, B_mean]
    grand_mean  = means.mean()
    gains       = grand_mean / (means + 1e-6)    # per-channel gain
    gains       = np.clip(gains, 0.1, 10.0)      # safety clamp — avoid extreme gains
    print(f'         AWB gains: R={gains[0]:.3f}  G={gains[1]:.3f}  B={gains[2]:.3f}')
    return np.clip(img * gains[np.newaxis, np.newaxis, :], 0.0, 1.0)

packed = gray_world_awb(packed)
print(f'[Step 4] AWB         — range: [{packed.min():.4f}, {packed.max():.4f}]')


# ── Step 5: Gamma correction (1/2.2) ─────────────────────────────────────────
# sRGB gamma applied AFTER AWB (as in a real ISP).
# Compresses the dynamic range, making the image look "normal" to a CNN
# that was pretrained on gamma-corrected images (ImageNet, COCO).

packed = np.power(packed, 1.0 / 2.2)
print(f'[Step 5] Gamma       — range: [{packed.min():.4f}, {packed.max():.4f}]')


# ── Step 6: Convert to uint8 BGR for inference_detector ──────────────────────
# inference_detector expects a standard OpenCV uint8 BGR ndarray.
# Our packed image is in RGB order → swap to BGR before passing.

packed_bgr_u8 = (packed[:, :, ::-1] * 255.0).clip(0, 255).astype(np.uint8)
print(f'[Step 6] To BGR u8   — shape: {packed_bgr_u8.shape}  range: [{packed_bgr_u8.min()}, {packed_bgr_u8.max()}]')

# Save the preprocessed image for visual inspection
preproc_path = os.path.join(OUT_DIR, 'day-00116_preprocessed.png')
cv2.imwrite(preproc_path, packed_bgr_u8)
print(f'         Saved preprocessed image to {preproc_path}')


# ── Step 7: Load COCO RTMDet-S and run inference ──────────────────────────────
# Passing an ndarray to inference_detector bypasses file loading.
# mmdet internally applies LoadImageFromNDArray + the rest of the test pipeline
# (Resize to 640, normalization), then runs the model.
# Returned boxes are in the packed-image coordinate space (928 × 1440).

model  = init_detector(CONFIG, CHECKPOINT, device='cuda:0')
result = inference_detector(model, packed_bgr_u8)

pred   = result.pred_instances
boxes  = pred.bboxes.cpu().numpy()
scores = pred.scores.cpu().numpy()
labels = pred.labels.cpu().numpy()

h_packed, w_packed = packed_bgr_u8.shape[:2]
print(f'\n[Step 7] Inference   — packed image: {h_packed} x {w_packed}')
print(f'         result.img_shape: {result.img_shape}')
print(f'         result.ori_shape: {result.ori_shape}')
if len(boxes) > 0:
    print(f'         Sample box (first): {boxes[0]}')


# ── Step 8: Filter + NMS ──────────────────────────────────────────────────────
from torchvision.ops import nms

keep   = (scores > SCORE_THR) & np.isin(labels, list(COCO_ROD_IDS.keys()))
boxes  = boxes[keep]
scores = scores[keep]
labels = labels[keep]

if len(boxes) > 0:
    keep_nms = nms(
        torch.tensor(boxes,  dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
        iou_threshold=0.45
    ).numpy()
    boxes  = boxes[keep_nms]
    scores = scores[keep_nms]
    labels = labels[keep_nms]

print(f'         Predictions after filter+NMS: {len(boxes)}')


# ── Step 9: Save .txt ─────────────────────────────────────────────────────────
txt_path = os.path.join(OUT_DIR, 'day-00116_preds.txt')
with open(txt_path, 'w') as f:
    f.write('Script 2 — RAW → fixed preprocessing → COCO RTMDet-S\n')
    f.write(f'RAW input:  {RAW_PATH}\n')
    f.write(f'Checkpoint: {CHECKPOINT}\n')
    f.write(f'Preprocessing: pack_bayer → p99_norm → gray_world_awb → gamma(1/2.2)\n')
    f.write(f'Score thr:  {SCORE_THR}\n')
    f.write(f'Packed image size: {h_packed} x {w_packed}\n')
    f.write(f'Total predictions: {len(boxes)}\n')
    f.write('─' * 60 + '\n')
    f.write(f'{"#":<5} {"class":<12} {"score":<8} {"x1":>6} {"y1":>6} {"x2":>6} {"y2":>6}\n')
    f.write('─' * 60 + '\n')
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        cls_name = COCO_ROD_IDS.get(int(label), f'cls_{label}')
        x1, y1, x2, y2 = box.astype(int)
        f.write(f'{i:<5} {cls_name:<12} {score:.4f}   {x1:>6} {y1:>6} {x2:>6} {y2:>6}\n')

print(f'\nSaved predictions to {txt_path}')


# ── Step 10: Save annotated image ─────────────────────────────────────────────
COLOURS = {
    'person':  (0,   255, 0),
    'bicycle': (255, 165, 0),
    'car':     (0,   0,   255),
    'train':   (255, 0,   255),
    'truck':   (0,   255, 255),
}

vis = packed_bgr_u8.copy()
for box, score, label in zip(boxes, scores, labels):
    x1, y1, x2, y2 = box.astype(int)
    cls_name = COCO_ROD_IDS.get(int(label), f'cls_{label}')
    colour   = COLOURS.get(cls_name, (255, 255, 255))
    cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
    cv2.putText(vis, f'{cls_name} {score:.2f}', (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

png_path = os.path.join(OUT_DIR, 'day-00116_preds.png')
cv2.imwrite(png_path, vis)
print(f'Saved annotated image to {png_path}')
