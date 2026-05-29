# ─────────────────────────────────────────────────────────────
# experiments/exp14_convgamma_k1.py
# Input:        Winner of {exp11, exp12} — update in_channels/PackBayer below
# Preprocessor: Winner of {exp13} — kernel_size=1
#               → ConvGamma    if exp13 (gain) loses to exp11/12 baseline
#               → ConvGammaGain if exp13 wins
# Detector:     RTMDet-S / Frozen
# Dataset:      ROD
# Question:     What is the best kernel size? (1 vs 3 vs 5)
#
# !! ACTION REQUIRED before running !!
#    1. Check exp13 result vs exp11/12:
#       - exp13 wins → change type to 'ConvGammaGain'
#       - exp13 loses → keep type as 'ConvGamma'
#    2. Update PackBayer out_channels + in_channels to match winner of {11,12}
# ─────────────────────────────────────────────────────────────

_base_ = [
    '../_base_/detectors/rtmdetS_frozen.py',
    '../_base_/datasets/rod_dataset.py',
    '../_base_/default_runtime.py',
]

exp_name = 'exp14'

auto_scale_lr = dict(enable=True, base_batch_size=64)

# ── Pipeline ──────────────────────────────────────────────────
train_pipeline = [
    dict(type='LoadRAWImageFromFile'),
    dict(type='NormaliseP99'),
    dict(type='PackBayer', out_channels=3),   # ← update to winner of {11, 12}
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadRAWImageFromFile'),
    dict(type='NormaliseP99'),
    dict(type='PackBayer', out_channels=3),   # ← update to winner of {11, 12}
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader   = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader  = dict(dataset=dict(pipeline=test_pipeline))

# ── Model ─────────────────────────────────────────────────────
model = dict(
    data_preprocessor=dict(
        type='RAWDetDataPreprocessor',
        preprocessor_cfg=dict(
            type='ConvGamma',       # ← change to 'ConvGammaGain' if exp13 wins
            in_channels=3,          # ← update to winner of {11, 12}
            out_channels=3,
            kernel_size=1,          # ← ONLY fixed difference from exp13
        ),
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=True,
        pad_size_divisor=32
    )
)

# ── WandB ─────────────────────────────────────────────────────
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='neuRAWns-mmdet-ROD-v2',
            name=exp_name,
            config=dict(
                input='packed_3ch',         # ← update if 4ch wins
                preprocessor='ConvGamma',   # ← update if exp13 wins
                gamma=1/2.2,
                in_channels=3,              # ← update if 4ch wins
                kernel_size=1,
                gain=False,                 # ← set True if exp13 wins
                detector='RTMDet-S',
                detector_frozen=True,
            )
        )
    )
]

visualizer = dict(vis_backends=vis_backends)

# ── Hooks ─────────────────────────────────────────────────────
custom_hooks = [
    dict(type='FreezeDetectorHook', debug_mode=False, check_updates=False, priority='VERY_HIGH'),
    dict(type='EarlyStoppingHook', monitor='coco/bbox_mAP', patience=10, rule='greater', min_delta=0.001),
    dict(type='SaveBatchImagesHook', save_dir='sample_images', experiment_name=exp_name, save_raw=True, save_preprocessed=True),
    dict(type='PreprocessorMonitorHook', log_every_n_steps=50)
]
