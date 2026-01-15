import torch
import cv2
import numpy as np
import sys

sys.path.insert(0, '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/neuRAWns_mmdet')

from mmdet.apis import init_detector, inference_detector
from mmengine.config import Config
from mmengine.dataset import Compose

# ============================================================
# PATHS
# ============================================================
std_config = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/mmdetection/mmdet/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
raw_config = 'debug_scripts/exp004_fixed.py'
checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
raw_file = '/cifs/Shares/Raw_Bayer_Datasets/ROD/raw/train/day-00000.raw'
isp_file = '/cifs/Shares/Raw_Bayer_Datasets/ROD/isp/train/day-00000.jpg'

device = 'cuda:0'

# ============================================================
# TEST: Feed FixedISP output to STANDARD detector
# ============================================================
print("="*60)
print("TEST: FixedISP output → Standard Detector (bypass RAWResNet)")
print("="*60)

# Step 1: Load RAW model just to get the preprocessor
cfg = Config.fromfile(raw_config)
raw_model = init_detector(cfg, checkpoint, device=device)
raw_model.eval()

# Step 2: Load RAW and preprocess
test_pipeline = Compose(cfg.test_pipeline)
data = dict(img_path=raw_file, img_id=0)
data = test_pipeline(data)

input_tensor = data['inputs']
if input_tensor.dim() == 3 and input_tensor.shape[0] == 1:
    input_tensor = input_tensor.unsqueeze(0)
input_tensor = input_tensor.to(device)

print(f"RAW input shape: {input_tensor.shape}")

# Step 3: Get preprocessed output from FixedISP
with torch.no_grad():
    preprocessed = raw_model.backbone.preprocessor(input_tensor)
    # Upsample like RAWResNet does
    preprocessed = torch.nn.functional.interpolate(preprocessed, scale_factor=2, mode='bilinear', align_corners=False)

print(f"Preprocessed shape: {preprocessed.shape}")
print(f"Preprocessed range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
print(f"Preprocessed mean: {preprocessed.mean():.3f}")

# Step 4: Denormalize to get RGB image [0, 255]
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
rgb_01 = preprocessed * std + mean
rgb_01 = torch.clamp(rgb_01, 0, 1)
rgb_255 = (rgb_01.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

print(f"RGB image shape: {rgb_255.shape}")

# Step 5: Save as temporary image and run standard detector on it
temp_path = '/tmp/fixedisp_output.jpg'
cv2.imwrite(temp_path, cv2.cvtColor(rgb_255, cv2.COLOR_RGB2BGR))
print(f"Saved temp image: {temp_path}")

# Step 6: Load STANDARD detector and run inference
std_model = init_detector(std_config, checkpoint, device=device)
std_model.eval()

results = inference_detector(std_model, temp_path)

scores = results.pred_instances.scores.cpu().numpy()
labels = results.pred_instances.labels.cpu().numpy()

# Filter to ROD classes
ROD_COCO_IDS = [0, 1, 2, 6, 7]
COCO_TO_ROD = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Car', 6: 'Tram', 7: 'Truck'}

rod_mask = np.isin(labels, ROD_COCO_IDS)
rod_scores = scores[rod_mask]
rod_labels = labels[rod_mask]

print(f"\n{'='*60}")
print("RESULTS: FixedISP output → Standard Detector")
print("="*60)
print(f"Total predictions: {len(scores)}")
print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
print(f"ROD predictions: {len(rod_scores)}")

if len(rod_scores) > 0:
    print(f"ROD score range: [{rod_scores.min():.3f}, {rod_scores.max():.3f}]")
    print(f"ROD detections >= 0.3: {(rod_scores >= 0.3).sum()}")
    print(f"ROD detections >= 0.5: {(rod_scores >= 0.5).sum()}")
    
    print("\nTop 5 ROD detections:")
    top_idx = np.argsort(rod_scores)[::-1][:5]
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. {COCO_TO_ROD[rod_labels[idx]]}: {rod_scores[idx]:.3f}")
else:
    print("⚠️ No ROD detections!")

# ============================================================
# COMPARISON with direct ISP image
# ============================================================
print(f"\n{'='*60}")
print("COMPARISON: Direct ISP image → Standard Detector")
print("="*60)

results_isp = inference_detector(std_model, isp_file)
scores_isp = results_isp.pred_instances.scores.cpu().numpy()
labels_isp = results_isp.pred_instances.labels.cpu().numpy()

rod_mask_isp = np.isin(labels_isp, ROD_COCO_IDS)
rod_scores_isp = scores_isp[rod_mask_isp]
rod_labels_isp = labels_isp[rod_mask_isp]

print(f"Total predictions: {len(scores_isp)}")
print(f"Score range: [{scores_isp.min():.3f}, {scores_isp.max():.3f}]")
print(f"ROD predictions: {len(rod_scores_isp)}")

if len(rod_scores_isp) > 0:
    print(f"ROD score range: [{rod_scores_isp.min():.3f}, {rod_scores_isp.max():.3f}]")
    print(f"ROD detections >= 0.3: {(rod_scores_isp >= 0.3).sum()}")
    print(f"ROD detections >= 0.5: {(rod_scores_isp >= 0.5).sum()}")
    
    print("\nTop 5 ROD detections:")
    top_idx = np.argsort(rod_scores_isp)[::-1][:5]
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. {COCO_TO_ROD[rod_labels_isp[idx]]}: {rod_scores_isp[idx]:.3f}")