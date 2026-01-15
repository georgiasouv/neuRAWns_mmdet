import torch
import cv2
import numpy as np
import sys
sys.path.insert(0, '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/neuRAWns_mmdet')

from mmdet.apis import init_detector
from mmengine.config import Config
from mmengine.dataset import Compose

# Configs
std_config = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/mmdetection/mmdet/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
raw_config = 'debug_scripts/exp004_fixed.py'
checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
raw_file = '/cifs/Shares/Raw_Bayer_Datasets/ROD/raw/train/day-00000.raw'

device = 'cuda:0'

# ============================================================
# PATH A: RAW → FixedISP → What tensor enters ResNet?
# ============================================================
print("="*60)
print("PATH A: RAW → RAWResNet pipeline")
print("="*60)

cfg_raw = Config.fromfile(raw_config)
model_raw = init_detector(cfg_raw, checkpoint, device=device)
model_raw.eval()

# Load RAW
test_pipeline = Compose(cfg_raw.test_pipeline)
data = dict(img_path=raw_file, img_id=0)
data = test_pipeline(data)

input_tensor = data['inputs']
if input_tensor.dim() == 3:
    input_tensor = input_tensor.unsqueeze(0)
input_tensor = input_tensor.to(device)

# Get what actually enters ResNet (after FixedISP + interpolate)
with torch.no_grad():
    preprocessed_a = model_raw.backbone.preprocessor(input_tensor)
    tensor_to_resnet_a = torch.nn.functional.interpolate(
        preprocessed_a, scale_factor=2, mode='bilinear', align_corners=False
    )

print(f"Tensor to ResNet (Path A):")
print(f"  Shape: {tensor_to_resnet_a.shape}")
print(f"  Range: [{tensor_to_resnet_a.min():.4f}, {tensor_to_resnet_a.max():.4f}]")
print(f"  Mean: {tensor_to_resnet_a.mean():.4f}")
print(f"  Std: {tensor_to_resnet_a.std():.4f}")
print(f"  R mean: {tensor_to_resnet_a[0,0].mean():.4f}")
print(f"  G mean: {tensor_to_resnet_a[0,1].mean():.4f}")
print(f"  B mean: {tensor_to_resnet_a[0,2].mean():.4f}")

# ============================================================
# PATH B: FixedISP output → Save JPG → Load → Standard detector preprocessing
# ============================================================
print("\n" + "="*60)
print("PATH B: FixedISP → JPG → Standard detector pipeline")
print("="*60)

# Save FixedISP output as JPG (same as before)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
rgb_01 = tensor_to_resnet_a * std + mean
rgb_01 = torch.clamp(rgb_01, 0, 1)
rgb_255 = (rgb_01.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

temp_path = '/tmp/fixedisp_output.jpg'
cv2.imwrite(temp_path, cv2.cvtColor(rgb_255, cv2.COLOR_RGB2BGR))

# Load standard model
model_std = init_detector(std_config, checkpoint, device=device)
model_std.eval()

# Load the JPG and see what tensor enters ResNet
img = cv2.imread(temp_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(device)

# Standard MMDet normalization (what DetDataPreprocessor does)
mean_std = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(device)
std_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(device)
tensor_to_resnet_b = (img_tensor - mean_std) / std_std
tensor_to_resnet_b = tensor_to_resnet_b.unsqueeze(0)

print(f"Tensor to ResNet (Path B):")
print(f"  Shape: {tensor_to_resnet_b.shape}")
print(f"  Range: [{tensor_to_resnet_b.min():.4f}, {tensor_to_resnet_b.max():.4f}]")
print(f"  Mean: {tensor_to_resnet_b.mean():.4f}")
print(f"  Std: {tensor_to_resnet_b.std():.4f}")
print(f"  R mean: {tensor_to_resnet_b[0,0].mean():.4f}")
print(f"  G mean: {tensor_to_resnet_b[0,1].mean():.4f}")
print(f"  B mean: {tensor_to_resnet_b[0,2].mean():.4f}")

# ============================================================
# COMPARE
# ============================================================
print("\n" + "="*60)
print("COMPARISON")
print("="*60)

diff = (tensor_to_resnet_a - tensor_to_resnet_b).abs()
print(f"Max absolute difference: {diff.max():.4f}")
print(f"Mean absolute difference: {diff.mean():.4f}")

print(f"\nMean comparison:")
print(f"  Path A mean: {tensor_to_resnet_a.mean():.4f}")
print(f"  Path B mean: {tensor_to_resnet_b.mean():.4f}")
print(f"  Difference: {(tensor_to_resnet_a.mean() - tensor_to_resnet_b.mean()):.4f}")


print("\n" + "="*60)
print("METADATA CHECK")
print("="*60)

# What metadata does RAW pipeline produce?
print("RAW pipeline data_samples metainfo:")
if 'data_samples' in data:
    print(f"  {data['data_samples'].metainfo}")
else:
    print("  No data_samples!")

# After DetDataPreprocessor
from mmdet.structures import DetDataSample
data_for_model = {
    'inputs': [input_tensor.squeeze(0)],
    'data_samples': [data['data_samples']] if 'data_samples' in data else [DetDataSample()]
}

preprocessor_output = model_raw.data_preprocessor(data_for_model)
print("\nAfter DetDataPreprocessor:")
print(f"  Input shape: {preprocessor_output['inputs'].shape}")
print(f"  data_samples metainfo: {preprocessor_output['data_samples'][0].metainfo}")

# What shape does the tensor actually have after RAWResNet preprocessing?
print(f"\nActual tensor shape after FixedISP + interpolate: {tensor_to_resnet_a.shape}")
print(f"img_shape in metadata: {preprocessor_output['data_samples'][0].metainfo.get('img_shape', 'NOT SET')}")