import torch
import cv2
import numpy as np
import sys

sys.path.insert(0, '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/neuRAWns_mmdet')

from mmdet.apis import init_detector, inference_detector
from mmengine.config import Config

# ============================================================
# Load ISP image and normalize EXACTLY like FixedISP does
# ============================================================
isp_file = '/cifs/Shares/Raw_Bayer_Datasets/ROD/isp/train/day-00000.jpg'

# Load ISP image
img = cv2.imread(isp_file)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0  # [0, 1] range

# Apply ImageNet normalization (same as FixedISP)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img_normalized = (img_tensor - mean) / std

print("ISP image with FixedISP-style normalization:")
print(f"  Shape: {img_normalized.shape}")
print(f"  Range: [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
print(f"  Mean: {img_normalized.mean():.3f}")
print(f"  R: mean={img_normalized[0].mean():.3f}")
print(f"  G: mean={img_normalized[1].mean():.3f}")
print(f"  B: mean={img_normalized[2].mean():.3f}")

# ============================================================
# Now compare to standard MMDet normalization
# ============================================================
# MMDet uses pixel values [0, 255], not [0, 1]
img_tensor_255 = torch.from_numpy(img_rgb).permute(2, 0, 1).float()  # [0, 255] range

mean_255 = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
std_255 = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
img_mmdet = (img_tensor_255 - mean_255) / std_255

print("\nISP image with MMDet-style normalization:")
print(f"  Shape: {img_mmdet.shape}")
print(f"  Range: [{img_mmdet.min():.3f}, {img_mmdet.max():.3f}]")
print(f"  Mean: {img_mmdet.mean():.3f}")
print(f"  R: mean={img_mmdet[0].mean():.3f}")
print(f"  G: mean={img_mmdet[1].mean():.3f}")
print(f"  B: mean={img_mmdet[2].mean():.3f}")

# ============================================================
# Check if they're equivalent
# ============================================================
print("\n" + "="*60)
print("Are they equivalent?")
print("="*60)
diff = (img_normalized - img_mmdet).abs()
print(f"Max absolute difference: {diff.max():.6f}")
print(f"Mean absolute difference: {diff.mean():.6f}")

if diff.max() < 0.001:
    print("✓ Normalizations are equivalent!")
else:
    print("✗ Normalizations differ!")