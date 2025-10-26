import torch
from mmengine.config import Config
from mmdet.registry import DATASETS

# Import the entire pipelines module to trigger registration
import datasets.pipelines.loading  # Add this line

import modules.raw_preprocessors
import modules.raw_backbones
import modules.hooks


cfg = Config.fromfile('configs/raw_experiments/exp002.py')


# Build dataset directly
dataset = DATASETS.build(cfg.train_dataloader.dataset)

print(f"\nDataset size: {len(dataset)}")
print("Checking first 200 samples...\n")

all_gt_labels = []

for i in range(min(200, len(dataset))):
    data = dataset[i]
    
    # Extract GT labels
    if hasattr(data['data_samples'], 'gt_instances'):
        labels = data['data_samples'].gt_instances.labels.tolist()
        all_gt_labels.extend(labels)

counter = Counter(all_gt_labels)

print("="*70)
print("GROUND TRUTH LABELS IN FIRST 200 TRAINING SAMPLES:")
print("="*70)
for class_id in range(5):
    count = counter.get(class_id, 0)
    class_name = cfg.classes[class_id] if hasattr(cfg, 'classes') else f"Class_{class_id}"
    print(f"  Class {class_id} ({class_name:12s}): {count:5d} instances")
print("="*70)

if counter.get(0, 0) == 0:
    print("\n⚠️⚠️⚠️ CLASS 0 (CAR) IS MISSING FROM GROUND TRUTH ⚠️⚠️⚠️\n")
else:
    print(f"\n✓ Class 0 present with {counter.get(0, 0)} instances\n")