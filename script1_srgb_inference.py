# ─────────────────────────────────────────────────────────────
# script1_srgb_inference.py
# Baseline: sRGB image → COCO pretrained RTMDet-S → predictions
# No custom preprocessor — pure detector sanity check
# ─────────────────────────────────────────────────────────────

import os
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector

# ── Paths ─────────────────────────────────────────────────────
CONFIG     = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/mmdetection/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'
CHECKPOINT = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/.cache/torch/hub/checkpoints/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'
IMAGE_PATH = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/data/ROD/yolo/srgb/images/test/day-00116.jpg'
OUT_DIR    = 'inference_results/script1_srgb'
SCORE_THR  = 0.3   # lowered from 0.3 — COCO on dashcam is harder

COCO_ROD_IDS = {0: 'person', 1: 'bicycle', 2: 'car', 6: 'train', 7: 'truck'}

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load model ────────────────────────────────────────────────
model = init_detector(CONFIG, CHECKPOINT, device='cuda:0')

# ── Run inference ─────────────────────────────────────────────
# Pass filepath directly — mmdet handles loading + resizing internally
# Boxes returned are already in original image coordinate space
result = inference_detector(model, IMAGE_PATH)

# ── Extract predictions ───────────────────────────────────────
pred   = result.pred_instances
boxes  = pred.bboxes.cpu().numpy()   # [N, 4] xyxy — original image coords
scores = pred.scores.cpu().numpy()   # [N]
labels = pred.labels.cpu().numpy()   # [N]

# Debug — understand coordinate spaces
img = cv2.imread(IMAGE_PATH)
orig_h, orig_w = img.shape[:2]
print(f'Original image shape: {orig_h} x {orig_w}')
print(f'result.img_shape:     {result.img_shape}')
print(f'result.ori_shape:     {result.ori_shape}')
if len(boxes) > 0:
    print(f'Sample box (first):   {boxes[0]}')

# Filter to ROD classes + score threshold
keep   = (scores > SCORE_THR) & np.isin(labels, list(COCO_ROD_IDS.keys()))
boxes  = boxes[keep]
scores = scores[keep]
labels = labels[keep]

from torchvision.ops import nms
import torch

if len(boxes) > 0:
    keep_nms = nms(
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
        iou_threshold=0.45   # lower = more aggressive overlap removal
    ).numpy()
    boxes  = boxes[keep_nms]
    scores = scores[keep_nms]
    labels = labels[keep_nms]
print(f'Predictions after filter: {len(boxes)}')

# ── Save .txt ─────────────────────────────────────────────────
txt_path = os.path.join(OUT_DIR, 'day-00116_preds.txt')
with open(txt_path, 'w') as f:
    f.write('Script 1 — sRGB inference (COCO pretrained, ROD classes only)\n')
    f.write(f'Image:      {IMAGE_PATH}\n')
    f.write(f'Checkpoint: {CHECKPOINT}\n')
    f.write(f'Score thr:  {SCORE_THR}\n')
    f.write(f'Total predictions: {len(boxes)}\n')
    f.write('─' * 60 + '\n')
    f.write(f'{"#":<5} {"class":<12} {"score":<8} {"x1":>6} {"y1":>6} {"x2":>6} {"y2":>6}\n')
    f.write('─' * 60 + '\n')
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        cls_name = COCO_ROD_IDS.get(int(label), f'cls_{label}')
        x1, y1, x2, y2 = box.astype(int)
        f.write(f'{i:<5} {cls_name:<12} {score:.4f}   {x1:>6} {y1:>6} {x2:>6} {y2:>6}\n')

print(f'Saved predictions to {txt_path}')

# ── Save .png ─────────────────────────────────────────────────
COLOURS = {
    'person':  (0,   255, 0),
    'bicycle': (255, 165, 0),
    'car':     (0,   0,   255),
    'train':   (255, 0,   255),
    'truck':   (0,   255, 255),
}

for box, score, label in zip(boxes, scores, labels):
    x1, y1, x2, y2 = box.astype(int)
    cls_name = COCO_ROD_IDS.get(int(label), f'cls_{label}')
    colour   = COLOURS.get(cls_name, (255, 255, 255))
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
    text = f'{cls_name} {score:.2f}'
    cv2.putText(img, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

png_path = os.path.join(OUT_DIR, 'day-00116_preds.png')
cv2.imwrite(png_path, img)
print(f'Saved image to {png_path}')