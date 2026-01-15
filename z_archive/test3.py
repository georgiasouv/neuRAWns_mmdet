import torch
import cv2
import numpy as np
import sys

sys.path.insert(0, '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/neuRAWns_mmdet')

from mmdet.apis import init_detector
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.structures import DetDataSample

# ============================================================
# TEST 1: What does standard detector see?
# ============================================================
print("="*60)
print("TEST 1: Standard Detector - What tensor enters ResNet?")
print("="*60)

std_config = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/mmdetection/mmdet/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
isp_file = '/cifs/Shares/Raw_Bayer_Datasets/ROD/isp/train/day-00000.jpg'

device = 'cuda:0'
std_model = init_detector(std_config, checkpoint, device=device)
std_model.eval()

# Load and preprocess ISP image the way standard detector does
img = cv2.imread(isp_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Manually apply what DetDataPreprocessor does
img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # HWC -> CHW

# Standard MMDet preprocessing: BGR->RGB already done, now normalize
# Default ImageNet normalization
mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
img_normalized = (img_tensor - mean) / std

print(f"Standard input shape: {img_normalized.shape}")
print(f"Standard input range: [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
print(f"Standard input mean: {img_normalized.mean():.3f}")
print(f"Standard input std: {img_normalized.std():.3f}")

# Per-channel stats
for i, name in enumerate(['R', 'G', 'B']):
    ch = img_normalized[i]
    print(f"  {name}: mean={ch.mean():.3f}, std={ch.std():.3f}")

# ============================================================
# TEST 2: What does your RAW pipeline produce?
# ============================================================
print("\n" + "="*60)
print("TEST 2: RAW Pipeline - What tensor enters ResNet?")
print("="*60)

raw_config = 'debug_scripts/exp004_fixed.py'
raw_file = '/cifs/Shares/Raw_Bayer_Datasets/ROD/raw/train/day-00000.raw'

cfg = Config.fromfile(raw_config)
raw_model = init_detector(cfg, checkpoint, device=device)
raw_model.eval()

# Load RAW
# Load RAW
test_pipeline = Compose(cfg.test_pipeline)
data = dict(img_path=raw_file, img_id=0)
data = test_pipeline(data)

input_tensor = data['inputs']
print(f"Raw tensor from pipeline: {input_tensor.shape}")

# Only add dimensions if needed
if input_tensor.dim() == 2:  # (H, W)
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
elif input_tensor.dim() == 3:  # (1, H, W) or (H, W, 1)
    if input_tensor.shape[0] == 1:
        input_tensor = input_tensor.unsqueeze(0)  # -> (1, 1, H, W)
    else:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
# If already (1, 1, H, W), do nothing

input_tensor = input_tensor.to(device)
print(f"Tensor shape for preprocessor: {input_tensor.shape}")
with torch.no_grad():
    raw_preprocessed = raw_model.backbone.preprocessor(input_tensor)

print(f"RAW preprocessed shape: {raw_preprocessed.shape}")
print(f"RAW preprocessed range: [{raw_preprocessed.min():.3f}, {raw_preprocessed.max():.3f}]")
print(f"RAW preprocessed mean: {raw_preprocessed.mean():.3f}")
print(f"RAW preprocessed std: {raw_preprocessed.std():.3f}")

# Per-channel stats
for i, name in enumerate(['R', 'G', 'B']):
    ch = raw_preprocessed[0, i]
    print(f"  {name}: mean={ch.mean():.3f}, std={ch.std():.3f}")

# ============================================================
# TEST 3: Direct comparison
# ============================================================
print("\n" + "="*60)
print("COMPARISON")
print("="*60)

print(f"\n{'Metric':<20} {'Standard':<15} {'RAW Pipeline':<15}")
print("-"*50)
print(f"{'Shape':<20} {str(img_normalized.shape):<15} {str(raw_preprocessed.shape):<15}")
print(f"{'Mean':<20} {img_normalized.mean():.3f}{'':>10} {raw_preprocessed.mean():.3f}")
print(f"{'Std':<20} {img_normalized.std():.3f}{'':>10} {raw_preprocessed.std():.3f}")
print(f"{'Min':<20} {img_normalized.min():.3f}{'':>10} {raw_preprocessed.min():.3f}")
print(f"{'Max':<20} {img_normalized.max():.3f}{'':>10} {raw_preprocessed.max():.3f}")