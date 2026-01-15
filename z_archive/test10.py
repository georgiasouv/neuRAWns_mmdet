import torch
import sys
sys.path.insert(0, '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/neuRAWns_mmdet')

from mmdet.apis import init_detector
from mmengine.config import Config

raw_config = 'debug_scripts/exp004_fixed.py'
checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# Load checkpoint directly
ckpt = torch.hub.load_state_dict_from_url(checkpoint)

print("="*60)
print("CHECKPOINT KEYS (first 20 backbone keys)")
print("="*60)
backbone_keys = [k for k in ckpt['state_dict'].keys() if k.startswith('backbone.')]
for k in backbone_keys[:20]:
    print(f"  {k}")

print("\n" + "="*60)
print("MODEL KEYS (first 20 backbone keys)")
print("="*60)

cfg_raw = Config.fromfile(raw_config)
model_raw = init_detector(cfg_raw, None, device='cpu')  # No checkpoint

model_keys = [k for k in model_raw.state_dict().keys() if k.startswith('backbone.')]
for k in model_keys[:20]:
    print(f"  {k}")