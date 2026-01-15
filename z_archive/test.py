import torch
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector

# Standard COCO config and weights
config_file = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/mmdetection/mmdet/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# Your ISP-processed image
rgb_path = '/cifs/Shares/Raw_Bayer_Datasets/ROD/isp/train/day-00000.jpg'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = init_detector(config_file, checkpoint_file, device=device)

results = inference_detector(model, rgb_path)

scores = results.pred_instances.scores.cpu().numpy()
labels = results.pred_instances.labels.cpu().numpy()
bboxes = results.pred_instances.bboxes.cpu().numpy()

# ============================================================
# ROD to COCO class mapping
# ============================================================
# ROD classes mapped to these COCO class IDs
ROD_COCO_IDS = [0, 1, 2, 6, 7]  # person, bicycle, car, train, truck

# COCO ID to ROD class name
COCO_TO_ROD_NAME = {
    0: 'Pedestrian',
    1: 'Cyclist', 
    2: 'Car',
    6: 'Tram',
    7: 'Truck'
}

# Full COCO classes for reference
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# ============================================================
# PRINT ALL DETECTIONS (before filtering)
# ============================================================
print("="*60)
print("ALL DETECTIONS (before ROD class filtering)")
print("="*60)
print(f"Total predictions: {len(scores)}")
print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
print(f"Detections >= 0.5: {(scores >= 0.5).sum()}")
print(f"Detections >= 0.3: {(scores >= 0.3).sum()}")

# ============================================================
# FILTER TO ROD CLASSES ONLY
# ============================================================
rod_mask = np.isin(labels, ROD_COCO_IDS)
rod_scores = scores[rod_mask]
rod_labels = labels[rod_mask]
rod_bboxes = bboxes[rod_mask]

print("\n" + "="*60)
print("ROD CLASSES ONLY (person, bicycle, car, train, truck)")
print("="*60)
print(f"ROD class predictions: {len(rod_scores)}")

if len(rod_scores) > 0:
    print(f"Score range: [{rod_scores.min():.3f}, {rod_scores.max():.3f}]")
    print(f"Detections >= 0.5: {(rod_scores >= 0.5).sum()}")
    print(f"Detections >= 0.3: {(rod_scores >= 0.3).sum()}")
    
    # Print per-class breakdown
    print("\nPer-class breakdown:")
    for coco_id in ROD_COCO_IDS:
        cls_mask = rod_labels == coco_id
        cls_scores = rod_scores[cls_mask]
        if len(cls_scores) > 0:
            rod_name = COCO_TO_ROD_NAME[coco_id]
            coco_name = COCO_CLASSES[coco_id]
            print(f"  {rod_name} ({coco_name}): {len(cls_scores)} detections, "
                  f"scores [{cls_scores.min():.3f}, {cls_scores.max():.3f}]")
    
    # Print top ROD detections
    print("\nTop 10 ROD class detections:")
    top_idx = np.argsort(rod_scores)[::-1][:10]
    for i, idx in enumerate(top_idx):
        rod_name = COCO_TO_ROD_NAME[rod_labels[idx]]
        coco_name = COCO_CLASSES[rod_labels[idx]]
        print(f"  {i+1}. {rod_name} ({coco_name}): {rod_scores[idx]:.3f}")
else:
    print("  No ROD class detections!")

# ============================================================
# VISUALISATION
# ============================================================
print("\n" + "="*60)
print("VISUALISATION")
print("="*60)

img = cv2.imread(rgb_path)
img_all = img.copy()      # All detections
img_rod = img.copy()      # ROD classes only

conf_threshold = 0.3

# Colours for ROD classes (BGR)
ROD_COLOURS = {
    0: (0, 255, 0),     # Pedestrian - Green
    1: (255, 0, 0),     # Cyclist - Blue
    2: (0, 0, 255),     # Car - Red
    6: (255, 255, 0),   # Tram - Cyan
    7: (0, 165, 255)    # Truck - Orange
}

# Draw ALL detections on img_all
count_all = 0
for bbox, score, label in zip(bboxes, scores, labels):
    if score < conf_threshold:
        continue
    count_all += 1
    x1, y1, x2, y2 = bbox.astype(int)
    
    # Use ROD colour if ROD class, else grey
    if label in ROD_COCO_IDS:
        colour = ROD_COLOURS[label]
        cls_name = f"{COCO_TO_ROD_NAME[label]}"
    else:
        colour = (128, 128, 128)  # Grey for non-ROD classes
        cls_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"class_{label}"
    
    cv2.rectangle(img_all, (x1, y1), (x2, y2), colour, 2)
    text = f"{cls_name}: {score:.2f}"
    cv2.putText(img_all, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

# Draw ROD-only detections on img_rod
count_rod = 0
for bbox, score, label in zip(rod_bboxes, rod_scores, rod_labels):
    if score < conf_threshold:
        continue
    count_rod += 1
    x1, y1, x2, y2 = bbox.astype(int)
    
    colour = ROD_COLOURS[label]
    rod_name = COCO_TO_ROD_NAME[label]
    
    cv2.rectangle(img_rod, (x1, y1), (x2, y2), colour, 2)
    
    # Label background
    text = f"{rod_name}: {score:.2f}"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img_rod, (x1, y1 - text_h - 10), (x1 + text_w, y1), colour, -1)
    cv2.putText(img_rod, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

print(f"All detections >= {conf_threshold}: {count_all}")
print(f"ROD detections >= {conf_threshold}: {count_rod}")

# Save outputs
cv2.imwrite('isp_all_detections.jpg', img_all)
cv2.imwrite('isp_rod_detections.jpg', img_rod)
print(f"\n✓ Saved: isp_all_detections.jpg (all COCO classes)")
print(f"✓ Saved: isp_rod_detections.jpg (ROD classes only)")

# Side-by-side: Original | All detections | ROD only
# Resize if needed for display
h, w = img.shape[:2]
combined = np.hstack([img, img_all, img_rod])
cv2.imwrite('isp_detection_comparison.jpg', combined)
print(f"✓ Saved: isp_detection_comparison.jpg (original | all | ROD only)")

# ============================================================
# LEGEND
# ============================================================
print("\n" + "="*60)
print("COLOUR LEGEND (for ROD classes)")
print("="*60)
print("  Pedestrian (person): Green")
print("  Cyclist (bicycle):   Blue")
print("  Car:                 Red")
print("  Tram (train):        Cyan")
print("  Truck:               Orange")
print("  Other COCO classes:  Grey")