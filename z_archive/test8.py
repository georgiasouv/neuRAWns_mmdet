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
# Load both models
# ============================================================
cfg_raw = Config.fromfile(raw_config)
model_raw = init_detector(cfg_raw, checkpoint, device=device)
model_raw.eval()

model_std = init_detector(std_config, checkpoint, device=device)
model_std.eval()

# ============================================================
# PATH A: Get tensor that would go to ResNet
# ============================================================
test_pipeline = Compose(cfg_raw.test_pipeline)
data = dict(img_path=raw_file, img_id=0)
data = test_pipeline(data)

input_tensor = data['inputs']
if input_tensor.dim() == 3:
    input_tensor = input_tensor.unsqueeze(0)
input_tensor = input_tensor.to(device)

with torch.no_grad():
    preprocessed_a = model_raw.backbone.preprocessor(input_tensor)
    tensor_a = torch.nn.functional.interpolate(
        preprocessed_a, scale_factor=2, mode='bilinear', align_corners=False
    )

# ============================================================
# PATH B: Get tensor from JPG path
# ============================================================
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
rgb_01 = tensor_a * std + mean
rgb_01 = torch.clamp(rgb_01, 0, 1)
rgb_255 = (rgb_01.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

temp_path = '/tmp/fixedisp_output.jpg'
cv2.imwrite(temp_path, cv2.cvtColor(rgb_255, cv2.COLOR_RGB2BGR))

img = cv2.imread(temp_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(device)

mean_std = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1).to(device)
std_std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1).to(device)
tensor_b = ((img_tensor - mean_std) / std_std).unsqueeze(0)

# ============================================================
# Compare ResNet features directly
# ============================================================
print("="*60)
print("RESNET FEATURE COMPARISON")
print("="*60)

with torch.no_grad():
    # Path A: Through RAWResNet's internal ResNet
    features_a = model_raw.backbone.resnet(tensor_a)
    
    # Path B: Through standard model's ResNet
    features_b = model_std.backbone(tensor_b)

print(f"\nNumber of feature levels: A={len(features_a)}, B={len(features_b)}")

for i in range(len(features_a)):
    fa = features_a[i]
    fb = features_b[i]
    
    diff = (fa - fb).abs()
    
    print(f"\nLevel {i}:")
    print(f"  Shape A: {fa.shape}, Shape B: {fb.shape}")
    print(f"  Mean A: {fa.mean():.4f}, Mean B: {fb.mean():.4f}")
    print(f"  Std A: {fa.std():.4f}, Std B: {fb.std():.4f}")
    print(f"  Max diff: {diff.max():.4f}, Mean diff: {diff.mean():.4f}")