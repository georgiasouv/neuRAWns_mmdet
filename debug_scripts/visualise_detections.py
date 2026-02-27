import torch
import numpy as np
import cv2
import sys
import os
from mmdet.structures import DetDataSample

# Add project root
sys.path.insert(0, '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/neuRAWns_mmdet')

from mmdet.apis import init_detector
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.structures import DetDataSample
import mmcv

# Configuration
config_file = 'debug_scripts/exp004_fixed.py'

checkpoint_file = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
raw_file = '/cifs/Shares/Raw_Bayer_Datasets/ROD/raw/train/day-00000.raw'
output_file = 'detection_visualization.jpg'

# ROD class names (matching your config)
ROD_CLASSES = ('person', 'bicycle', 'car', 'train', 'truck')

print("="*80)
print("STEP 1: Initialize Model and Test Pipeline")
print("="*80)

# Load config
cfg = Config.fromfile(config_file)
print("\n[DEBUG] Test Pipeline Operators:")
for op in cfg.test_pipeline:
    print(" -", op)


# Build model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = init_detector(cfg, checkpoint_file, device=device)
model.eval()

print(f"✓ Model loaded on {device}")

# Build test pipeline from config
test_pipeline = Compose(cfg.test_pipeline)
print(f"✓ Test pipeline built with {len(cfg.test_pipeline)} transforms")

print("\n" + "="*80)
print("STEP 2: Load RAW Image Using Custom Pipeline")
print("="*80)

# Prepare data dict (mimics what dataloader provides)
data = dict(
    img_path=raw_file,
    img_id=0,
)

# Apply test pipeline (this calls your LoadRAWImageFromFile)
data = test_pipeline(data)

print(f"✓ RAW image loaded via pipeline")
print(f"  Image shape: {data['inputs'].shape}")
print(f"  Image range: [{data['inputs'].min():.2f}, {data['inputs'].max():.2f}]")


print("\n" + "="*80)
print("STEP 3: Run Inference")
print("="*80)

# The test pipeline outputs a tensor, but we need to add channel dim if missing
input_tensor = data['inputs']

# Check if channel dimension is missing
if input_tensor.dim() == 3:
    if input_tensor.shape[0] == 1:
        input_tensor = input_tensor.unsqueeze(1)
    else:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
elif input_tensor.dim() == 2:
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

print(f"  Prepared tensor shape: {input_tensor.shape}")

# Update the data dict
data['inputs'] = input_tensor.to(device)

# Make sure data_samples is a list
if 'data_samples' in data and not isinstance(data['data_samples'], list):
    data['data_samples'] = [data['data_samples']]
elif 'data_samples' not in data:
    data_sample = DetDataSample()
    data['data_samples'] = [data_sample]
    
print("\nMETAINFO (BEFORE INFERENCE):")
print(data['data_samples'][0].metainfo)

# Run inference
with torch.no_grad():
    print(f"\n[DEBUG] scale_factor investigation:")
    print(f"  Type: {type(data['data_samples'][0].scale_factor)}")
    print(f"  Value: {data['data_samples'][0].scale_factor}")
    print(f"  Shape: {data['data_samples'][0].scale_factor.shape if hasattr(data['data_samples'][0].scale_factor, 'shape') else 'no shape attr'}")
    print(f"  Length: {len(data['data_samples'][0].scale_factor)}")
    results = model.test_step(data)

result = results[0]
pred_instances = result.pred_instances

# Extract ALL predictions (before filtering)
all_labels = pred_instances.labels.cpu().numpy()
all_scores = pred_instances.scores.cpu().numpy()
all_bboxes = pred_instances.bboxes.cpu().numpy()

print(f"✓ Inference complete")
print(f"  Total raw predictions: {len(all_labels)}")

if len(all_labels) > 0:
    print(f"  Score range: [{all_scores.min():.3f}, {all_scores.max():.3f}]")
    print(f"  Classes predicted (raw): {np.unique(all_labels)}")
    
    # Show score distribution
    for thresh in [0.05, 0.1, 0.2, 0.3, 0.5]:
        count = (all_scores >= thresh).sum()
        print(f"    Predictions >= {thresh}: {count}")
else:
    print(f"  ⚠️  WARNING: No predictions at all!")

# Filter by confidence threshold
confidence_threshold = 0.3  # ← Lower threshold to see more detections
mask = all_scores >= confidence_threshold
labels = all_labels[mask]
scores = all_scores[mask]
bboxes = all_bboxes[mask]


print("RAW BBOXES (first 10):", bboxes[:10])
print("RAW SCORES (first 10):", scores[:10])
print("RAW LABELS (unique):", np.unique(labels))

print(f"  Detections above {confidence_threshold}: {len(labels)}")

if len(labels) > 0:
    print(f"  Filtered classes: {np.unique(labels)}")
    for cls_id in np.unique(labels):
        cls_name = ROD_CLASSES[cls_id] if cls_id < len(ROD_CLASSES) else f"class_{cls_id}"
        cls_count = (labels == cls_id).sum()
        cls_scores = scores[labels == cls_id]
        print(f"    {cls_name}: {cls_count} detections, scores [{cls_scores.min():.3f}, {cls_scores.max():.3f}]")


# print(f"  Detections above {confidence_threshold}: {len(labels)}")
print("\n[DEBUG] Checking what detector actually sees:")

with torch.no_grad():
    # 1) Make sure the tensor is on the same device as the model
    #    Use the backbone.preprocessor's device as the source of truth
    preproc_device = next(model.backbone.preprocessor.parameters(), torch.zeros(1)).device
    test_tensor = input_tensor.to(preproc_device)

    # 2) Pass through backbone preprocessing
    preprocessed = model.backbone.preprocessor(test_tensor)

    print(f"After full preprocessing (what detector sees):")
    print(f"  Shape: {preprocessed.shape}")
    print(f"  Range: [{preprocessed.min():.4f}, {preprocessed.max():.4f}]")
    print(f"  Mean: {preprocessed.mean():.4f}")
    print(f"  Std: {preprocessed.std():.4f}")

    # 3) Move to CPU for visualisation – avoids device mismatch entirely
    preprocessed_cpu = preprocessed.detach().cpu()

    # 4) Denormalize to see the actual RGB, on CPU
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    rgb_denorm = preprocessed_cpu * std + mean

    print(f"After denorm (actual RGB before ImageNet norm):")
    print(f"  Range: [{rgb_denorm.min():.4f}, {rgb_denorm.max():.4f}]")
    print(f"  Mean: {rgb_denorm.mean():.4f}")

    # 5) Save this image
    debug_img = rgb_denorm.squeeze(0).numpy().transpose(1, 2, 0)
    debug_img = (np.clip(debug_img, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite('debug_detector_sees.jpg', cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    print(f"  ✓ Saved: debug_detector_sees.jpg (this is what the detector actually processes)")


print("\n" + "="*80)
print("STEP 4: Create Visualization - USING RESIZED RAW")
print("="*80)

resized_raw_tensor = input_tensor  # [1, 1, H, W]

with torch.no_grad():
    # Ensure correct device
    preproc_device = next(model.backbone.preprocessor.parameters(), torch.zeros(1)).device
    resized_raw_tensor = resized_raw_tensor.to(preproc_device)

    # Preprocess on same device
    rgb_tensor = model.backbone.preprocessor(resized_raw_tensor)

    # Move to CPU immediately for visualisation
    rgb_tensor_cpu = rgb_tensor.detach().cpu()

    # Denormalise on CPU only
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    rgb_tensor_cpu = rgb_tensor_cpu * std + mean
    rgb_tensor_cpu = torch.clamp(rgb_tensor_cpu, 0, 1)

    # Convert to image
    rgb_img = rgb_tensor_cpu.squeeze(0).numpy().transpose(1, 2, 0)
    rgb_img = (rgb_img * 255).astype(np.uint8)

cv2.imwrite("detector_actually_sees_resized.jpg",
            cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
print("✓ Saved what detector ACTUALLY sees: detector_actually_sees_resized.jpg")


# ----------------------------------------------------------------------
# STEP 5: DRAW BOUNDING BOXES ON THE ISP OUTPUT (DEBUG RGB)
# ----------------------------------------------------------------------

print("\nSTEP 5: Drawing bounding boxes...")

# Use the *denormalised* RGB image you already saved
vis_img = debug_img.copy()  # debug_img from earlier, uint8 HxWx3 RGB
scaled_bboxes = bboxes / 2.0 
for bbox, score, label in zip(scaled_bboxes, scores, labels):
    x1, y1, x2, y2 = bbox.astype(int)

    # Draw rectangle
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw label + score
    class_name = ROD_CLASSES[label] if label < len(ROD_CLASSES) else f"class_{label}"
    text = f"{class_name}: {score:.2f}"
    cv2.putText(vis_img, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

out_path = "raw_detection_overlay.jpg"
cv2.imwrite(out_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
print(f"✓ Saved visualised detections to {out_path}")
