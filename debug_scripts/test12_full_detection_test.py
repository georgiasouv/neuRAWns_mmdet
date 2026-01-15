import torch
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/neuRAWns_mmdet')

from mmdet.apis import init_detector
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.structures import DetDataSample

# Paths
raw_config = 'debug_scripts/exp004_fixed.py'
checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
raw_file = '/cifs/Shares/Raw_Bayer_Datasets/ROD/raw/train/day-00000.raw'

# ROD class mapping
ROD_COCO_IDS = [0, 1, 2, 6, 7]
COCO_TO_ROD_NAME = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car', 6: 'Tram', 7: 'Truck'}
ROD_COLOURS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255), 6: (255, 255, 0), 7: (0, 165, 255)}

device = 'cuda:0'

print("="*60)
print("FINAL TEST: RAW → FixedISP → RAWResNet → Detector")
print("="*60)

# Load model
cfg = Config.fromfile(raw_config)
model = init_detector(cfg, checkpoint, device=device)
model.eval()

# Load RAW
test_pipeline = Compose(cfg.test_pipeline)
data = dict(img_path=raw_file, img_id=0)
data = test_pipeline(data)

input_tensor = data['inputs']
if input_tensor.dim() == 3:
    input_tensor = input_tensor.unsqueeze(0)
input_tensor = input_tensor.to(device)

# Prepare data_samples
data['inputs'] = input_tensor
if 'data_samples' not in data or data['data_samples'] is None:
    data['data_samples'] = [DetDataSample()]
elif not isinstance(data['data_samples'], list):
    data['data_samples'] = [data['data_samples']]

# Run inference
with torch.no_grad():
    results = model.test_step(data)

result = results[0]
scores = result.pred_instances.scores.cpu().numpy()
labels = result.pred_instances.labels.cpu().numpy()
bboxes = result.pred_instances.bboxes.cpu().numpy()

# Filter to ROD classes
rod_mask = np.isin(labels, ROD_COCO_IDS)
rod_scores = scores[rod_mask]
rod_labels = labels[rod_mask]
rod_bboxes = bboxes[rod_mask]

print(f"\nResults:")
print(f"  Total predictions: {len(scores)}")
print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
print(f"  ROD class predictions: {len(rod_scores)}")

if len(rod_scores) > 0:
    print(f"  ROD score range: [{rod_scores.min():.3f}, {rod_scores.max():.3f}]")
    print(f"  ROD detections >= 0.3: {(rod_scores >= 0.3).sum()}")
    print(f"  ROD detections >= 0.5: {(rod_scores >= 0.5).sum()}")
    
    print("\n  Top 5 ROD detections:")
    top_idx = np.argsort(rod_scores)[::-1][:5]
    for i, idx in enumerate(top_idx):
        print(f"    {i+1}. {COCO_TO_ROD_NAME[rod_labels[idx]]}: {rod_scores[idx]:.3f}")
else:
    print("  ⚠️ No ROD class detections!")

# Visualize
print("\n" + "="*60)
print("VISUALIZATION")
print("="*60)

with torch.no_grad():
    preprocessed = model.backbone.preprocessor(input_tensor)
    preprocessed = torch.nn.functional.interpolate(preprocessed, scale_factor=2, mode='bilinear', align_corners=False)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    rgb = preprocessed * std + mean
    rgb = torch.clamp(rgb, 0, 1)
    rgb_img = (rgb.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

# Draw detections
conf_threshold = 0.3
vis_img = rgb_img.copy()
count = 0

for bbox, score, label in zip(rod_bboxes, rod_scores, rod_labels):
    if score < conf_threshold:
        continue
    count += 1
    x1, y1, x2, y2 = bbox.astype(int)
    colour = ROD_COLOURS[label]
    rod_name = COCO_TO_ROD_NAME[label]
    
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), colour, 2)
    text = f"{rod_name}: {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(vis_img, (x1, y1 - th - 10), (x1 + tw, y1), colour, -1)
    cv2.putText(vis_img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

print(f"Detections drawn: {count}")

folder = './visualisation'
os.makedirs(folder, exist_ok=True)
cv2.imwrite(os.path.join(folder, 'final_raw_detections.jpg'), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
print(f"✓ Saved: {folder}/final_raw_detections.jpg")