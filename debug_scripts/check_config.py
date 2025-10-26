from mmengine.config import Config
import os

config_file = 'configs/baselines/exp001_faster_rcnn_r50_fpn.py'
cfg = Config.fromfile(config_file)

print(f"Max epochs: {cfg.train_cfg.get('max_epochs', 'NOT SET')}")
print(f"Val interval: {cfg.train_cfg.get('val_interval', 'NOT SET')}")
print("OPTIMIZER CONFIG")
print(f"Type: {cfg.optim_wrapper.optimizer.type}")
print(f"LR: {cfg.optim_wrapper.optimizer.lr}")
print(f"Momentum: {cfg.optim_wrapper.optimizer.get('momentum', 'N/A')}")
print(f"Weight decay: {cfg.optim_wrapper.optimizer.weight_decay}")
print("CHECKPOINT CONFIG")
print("="*70)
if hasattr(cfg, 'default_hooks') and 'checkpoint' in cfg.default_hooks:
    ckpt = cfg.default_hooks.checkpoint
    print(f"Interval: {ckpt.get('interval', 'NOT SET')}")
    print(f"Max keep: {ckpt.get('max_keep_ckpts', 'NOT SET')}")
    print(f"Save best: {ckpt.get('save_best', 'NOT SET')}")
else:
    print("Using default checkpoint config")
print("WORK DIRECTORY")
print(f"Work dir: {cfg.get('work_dir', 'NOT SET - will use default')}")
print("WANDB CONFIG")
print("="*70)
if hasattr(cfg, 'vis_backends'):
    for backend in cfg.vis_backends:
        if backend['type'] == 'WandbVisBackend':
            print(f"WandB enabled: YES")
            print(f"Project: {backend['init_kwargs']['project']}")
            print(f"Name: {backend['init_kwargs']['name']}")
            break
    else:
        print("WandB enabled: NO")
else:
    print("No visualization backends configured")
print("LEARNING RATE SCHEDULE")
if hasattr(cfg, 'param_scheduler'):
    for scheduler in cfg.param_scheduler:
        print(f"- {scheduler['type']}: {scheduler}")
else:
    print("No LR scheduler found")