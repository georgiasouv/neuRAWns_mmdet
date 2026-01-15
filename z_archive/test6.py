import torch
import numpy as np
import sys
sys.path.insert(0, '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/Desktop/neuRAWns_mmdet')

from mmdet.apis import init_detector
from mmengine.config import Config
from mmengine.dataset import Compose
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData

raw_config = 'debug_scripts/exp004_fixed.py'
checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
raw_file = '/cifs/Shares/Raw_Bayer_Datasets/ROD/raw/train/day-00000.raw'

device = 'cuda:0'

# Load model
cfg = Config.fromfile(raw_config)
model = init_detector(cfg, checkpoint, device=device)
model.eval()

# Load RAW through pipeline (this sets up metadata correctly)
test_pipeline = Compose(cfg.test_pipeline)
data = dict(img_path=raw_file, img_id=0)
data = test_pipeline(data)

print(f"Pipeline output keys: {data.keys()}")
print(f"Input tensor shape: {data['inputs'].shape}")

if 'data_samples' in data:
    ds = data['data_samples']
    print(f"data_samples metainfo: {ds.metainfo}")

# ============================================================
# Check what DetDataPreprocessor does
# ============================================================
print("\n" + "="*60)
print("What does DetDataPreprocessor do to the input?")
print("="*60)

input_tensor = data['inputs']
if input_tensor.dim() == 3:
    input_tensor = input_tensor.unsqueeze(0)

print(f"Before DetDataPreprocessor: {input_tensor.shape}, range: [{input_tensor.min():.0f}, {input_tensor.max():.0f}]")

# Manually call DetDataPreprocessor
preprocessor_output = model.data_preprocessor({
    'inputs': [input_tensor.squeeze(0)],  # List of tensors
    'data_samples': [data['data_samples']] if 'data_samples' in data else [DetDataSample()]
})

processed_inputs = preprocessor_output['inputs']
processed_samples = preprocessor_output['data_samples']

print(f"After DetDataPreprocessor: {processed_inputs.shape}, range: [{processed_inputs.min():.0f}, {processed_inputs.max():.0f}]")
print(f"Processed data_samples metainfo: {processed_samples[0].metainfo}")

# ============================================================
# Now trace through backbone manually
# ============================================================
print("\n" + "="*60)
print("Backbone processing")
print("="*60)

with torch.no_grad():
    # What RAWResNet receives
    backbone_input = processed_inputs
    print(f"Backbone input: {backbone_input.shape}")
    
    # Through preprocessor only
    preprocessed = model.backbone.preprocessor(backbone_input)
    print(f"After FixedISP: {preprocessed.shape}")
    
    # After interpolate
    upsampled = torch.nn.functional.interpolate(preprocessed, scale_factor=2, mode='bilinear', align_corners=False)
    print(f"After interpolate: {upsampled.shape}")