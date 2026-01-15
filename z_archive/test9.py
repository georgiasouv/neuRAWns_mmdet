import torch
import sys
sys.path.insert(0, '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/neuRAWns_mmdet')

from mmdet.apis import init_detector
from mmengine.config import Config

std_config = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/mmdetection/mmdet/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
raw_config = 'debug_scripts/exp004_fixed.py'
checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

device = 'cuda:0'

cfg_raw = Config.fromfile(raw_config)
model_raw = init_detector(cfg_raw, checkpoint, device=device)

model_std = init_detector(std_config, checkpoint, device=device)

print("="*60)
print("WEIGHT COMPARISON")
print("="*60)

# Compare first conv layer
raw_conv1 = model_raw.backbone.resnet.conv1.weight
std_conv1 = model_std.backbone.conv1.weight

print(f"\nconv1 weights:")
print(f"  RAW shape: {raw_conv1.shape}, STD shape: {std_conv1.shape}")
print(f"  RAW mean: {raw_conv1.mean():.6f}, STD mean: {std_conv1.mean():.6f}")
print(f"  Max diff: {(raw_conv1 - std_conv1).abs().max():.6f}")
print(f"  Are identical: {torch.allclose(raw_conv1, std_conv1)}")

# Compare a deeper layer
raw_layer1 = model_raw.backbone.resnet.layer1[0].conv1.weight
std_layer1 = model_std.backbone.layer1[0].conv1.weight

print(f"\nlayer1[0].conv1 weights:")
print(f"  RAW shape: {raw_layer1.shape}, STD shape: {std_layer1.shape}")
print(f"  RAW mean: {raw_layer1.mean():.6f}, STD mean: {std_layer1.mean():.6f}")
print(f"  Max diff: {(raw_layer1 - std_layer1).abs().max():.6f}")
print(f"  Are identical: {torch.allclose(raw_layer1, std_layer1)}")

# Check bn layers too
raw_bn1 = model_raw.backbone.resnet.bn1.running_mean
std_bn1 = model_std.backbone.bn1.running_mean

print(f"\nbn1 running_mean:")
print(f"  RAW: {raw_bn1[:5]}")
print(f"  STD: {std_bn1[:5]}")
print(f"  Are identical: {torch.allclose(raw_bn1, std_bn1)}")