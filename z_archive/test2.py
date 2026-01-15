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

# ============================================================
# PATHS
# ============================================================
raw_config = 'debug_scripts/exp004_fixed.py'
checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
raw_file = '/cifs/Shares/Raw_Bayer_Datasets/ROD/raw/train/day-00000.raw'
isp_file = '/cifs/Shares/Raw_Bayer_Datasets/ROD/isp/train/day-00000.jpg'

# ROD class mapping
ROD_COCO_IDS = [0, 1, 2, 6, 7]
COCO_TO_ROD_NAME = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car', 6: 'Tram', 7: 'Truck'}
ROD_COLOURS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255), 6: (255, 255, 0), 7: (0, 165, 255)}

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ============================================================
# PART 1: Load RAW model and run inference
# ============================================================
print("="*60)
print("RAW PIPELINE TEST")
print("="*60)

cfg = Config.fromfile(raw_config)
model = init_detector(cfg, checkpoint_file, device=device)
model.eval()

# Load RAW through pipeline
test_pipeline = Compose(cfg.test_pipeline)
data = dict(img_path=raw_file, img_id=0)
data = test_pipeline(data)

# Prepare tensor
input_tensor = data['inputs']
if input_tensor.dim() == 3:
    input_tensor = input_tensor.unsqueeze(0)
if input_tensor.dim() == 3:
    input_tensor = input_tensor.unsqueeze(0)
    
# Ensure shape is [B, 1, H, W]
if input_tensor.shape[1] != 1:
    input_tensor = input_tensor.unsqueeze(1)

print(f"Input tensor shape: {input_tensor.shape}")

data['inputs'] = input_tensor.to(device)

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

print(f"\nRAW Pipeline Results:")
print(f"  Total predictions: {len(scores)}")
print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
print(f"  ROD class predictions: {len(rod_scores)}")

if len(rod_scores) > 0:
    print(f"  ROD score range: [{rod_scores.min():.3f}, {rod_scores.max():.3f}]")
    print(f"  ROD detections >= 0.3: {(rod_scores >= 0.3).sum()}")
    print(f"  ROD detections >= 0.5: {(rod_scores >= 0.5).sum()}")
    
    print("\n  Top ROD detections:")
    top_idx = np.argsort(rod_scores)[::-1][:5]
    for i, idx in enumerate(top_idx):
        print(f"    {i+1}. {COCO_TO_ROD_NAME[rod_labels[idx]]}: {rod_scores[idx]:.3f}")
else:
    print("  ⚠️ No ROD class detections!")

# ============================================================
# PART 2: Get preprocessed image for visualization
# ============================================================
print("\n" + "="*60)
print("PREPROCESSING COMPARISON")
print("="*60)

with torch.no_grad():
    # Get what FixedISP produces
    raw_tensor = input_tensor.to(device)
    preprocessed = model.backbone.preprocessor(raw_tensor)
    
    # Denormalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    rgb_denorm = preprocessed * std + mean
    rgb_denorm = torch.clamp(rgb_denorm, 0, 1)
    
    # Convert to numpy image
    raw_rgb = rgb_denorm.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    raw_rgb = (raw_rgb * 255).astype(np.uint8)

print(f"Preprocessed shape: {preprocessed.shape}")
print(f"Preprocessed range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
print(f"Preprocessed mean: {preprocessed.mean():.3f}")

# Load ISP image for comparison
isp_img = cv2.imread(isp_file)
isp_img = cv2.cvtColor(isp_img, cv2.COLOR_BGR2RGB)

print(f"\nFixedISP output size: {raw_rgb.shape}")
print(f"ISP image size: {isp_img.shape}")

# ============================================================
# PART 3: Visualize detections on preprocessed RAW
# ============================================================
print("\n" + "="*60)
print("VISUALIZATION")
print("="*60)

conf_threshold = 0.3

# Draw on preprocessed RAW output
raw_vis = raw_rgb.copy()
count = 0
for bbox, score, label in zip(rod_bboxes, rod_scores, rod_labels):
    if score < conf_threshold:
        continue
    count += 1
    
    # Scale bboxes if needed (preprocessed is half resolution)
    # Bboxes are in preprocessed image coordinates
    x1, y1, x2, y2 = (bbox / 2.0).astype(int)  # Adjust if needed
    
    colour = ROD_COLOURS[label]
    rod_name = COCO_TO_ROD_NAME[label]
    
    cv2.rectangle(raw_vis, (x1, y1), (x2, y2), colour, 2)
    text = f"{rod_name}: {score:.2f}"
    cv2.putText(raw_vis, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

print(f"ROD detections drawn: {count}")

folder = './visualisation'
os.makedirs(folder, exist_ok=True)
# Save outputs
cv2.imwrite(os.path.join(folder, 'raw_pipeline_detections.jpg'), cv2.cvtColor(raw_vis, cv2.COLOR_RGB2BGR))
print(f"✓ Saved: raw_pipeline_detections.jpg")

# Save preprocessed image without detections
cv2.imwrite(os.path.join(folder, 'raw_preprocessed_output.jpg'), cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR))
print(f"✓ Saved: raw_preprocessed_output.jpg")

# Side by side: ISP (resized) vs RAW preprocessed
isp_resized = cv2.resize(isp_img, (raw_rgb.shape[1], raw_rgb.shape[0]))
comparison = np.hstack([isp_resized, raw_rgb])
cv2.imwrite(os.path.join(folder, 'isp_vs_raw_preprocessed.jpg'), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
print(f"✓ Saved: isp_vs_raw_preprocessed.jpg (ISP left | RAW preprocessed right)")