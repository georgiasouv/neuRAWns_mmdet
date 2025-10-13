#!/usr/bin/env python
"""
Verify Freezing Script
Tests that detector freezing will work correctly before starting training.
"""

import sys
import torch
from mmengine.config import Config
from mmdet.registry import MODELS
import modules.raw_backbones
import modules.hooks
import datasets.pipelines
import mmdet.models
from mmdet.models.data_preprocessors import DetDataPreprocessor

def verify_freezing(config_path):
    """
    Verify that freezing will work with the given config.
    
    Args:
        config_path: Path to the config file
    """
    print(f"\n{'='*70}")
    print("VERIFYING DETECTOR FREEZING")
    print(f"{'='*70}")
    print(f"Config: {config_path}\n")
    
    # Load config
    cfg = Config.fromfile(config_path)
    
    # Build model
    print("Building model...")
    model = MODELS.build(cfg.model)
    
    # Simulate hook logic: freeze everything, unfreeze preprocessor
    print("Simulating freeze logic...")
    for param in model.parameters():
        param.requires_grad = False
    
    if hasattr(model.backbone, 'preprocessor'):
        for param in model.backbone.preprocessor.parameters():
            param.requires_grad = True
    else:
        print("\n❌ ERROR: No preprocessor found in model.backbone!")
        print("Check your backbone configuration.")
        sys.exit(1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    percentage = 100 * trainable_params / total_params
    
    # Display results
    print(f"\n{'='*70}")
    print("PARAMETER ANALYSIS")
    print(f"{'='*70}")
    print(f"Total parameters:     {total_params:>15,}")
    print(f"Trainable parameters: {trainable_params:>15,}")
    print(f"Frozen parameters:    {frozen_params:>15,}")
    print(f"Percentage trainable: {percentage:>15.6f}%")
    print(f"{'='*70}\n")
    
    # List trainable modules
    print("Trainable modules:")
    trainable_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  - {name}: {param.numel():,} params")
            trainable_count += 1
    
    if trainable_count == 0:
        print("  (none)")
    
    print(f"\n{'='*70}")
    
    # Validation
    if trainable_params == 0:
        print("❌ FAIL: No trainable parameters!")
        print("The preprocessor was not found or has no parameters.")
        sys.exit(1)
    elif trainable_params > 50000:
        print(f"❌ FAIL: Too many trainable parameters ({trainable_params:,})")
        print("Expected < 50,000 parameters (just the preprocessor).")
        print("Some detector components may not be frozen.")
        sys.exit(1)
    elif percentage > 0.2:
        print(f"⚠️  WARNING: {percentage:.2f}% trainable seems high")
        print("Expected < 0.1% for typical detectors.")
        print("This might be okay if you have a small detector.")
    else:
        print("✓ SUCCESS: Freezing configuration is correct!")
        print(f"Only {trainable_params:,} parameters will be trained.")
        print("Ready to start training.")
    
    print(f"{'='*70}\n")
    
    return trainable_params < 50000

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python debug_scripts/verify_freezing.py <config_path>")
        print("Example: python debug_scripts/verify_freezing.py configs/raw_experiments/exp002.py")
        sys.exit(1)
    
    config_path = sys.argv[1]
    success = verify_freezing(config_path)
    sys.exit(0 if success else 1)