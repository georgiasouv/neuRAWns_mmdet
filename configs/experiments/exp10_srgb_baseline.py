# ─────────────────────────────────────────────────────────────
# experiments/exp10_srgb_baseline.py
# Input:        sRGB (standard 3ch JPEG/PNG)
# Preprocessor: None (Full ISP already applied)
# Detector:     RTMDet-S / standard (NOT frozen — upper reference)
# Dataset:      ROD (sRGB split)
# Purpose:      Upper reference — best possible with clean sRGB input
# ─────────────────────────────────────────────────────────────

_base_ = [
    '../_base_/detectors/rtmdetS_frozen.py',
    '../_base_/datasets/rod_dataset.py',
    '../_base_/default_runtime.py',
]

exp_name = 'exp10'

auto_scale_lr = dict(enable=True, base_batch_size=64)

# ── Pipeline ──────────────────────────────────────────────────
# Standard pipeline: loads sRGB images, no RAW-specific steps
train_pipeline = [
    dict(type='LoadImageFromFile'),          # standard mmdet loader, not LoadRAWImageFromFile
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs')
]

# ── Override dataloader pipelines ─────────────────────────────
# IMPORTANT: also point to sRGB image paths, not raw/images/
train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline,
        ann_file='srgb/json_srgb_coco_mapped/train.json',   # ← update to your sRGB split path
        data_prefix=dict(img='srgb/images/train/'),          # ← update to your sRGB split path
    )
)
val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        ann_file='srgb/json_srgb_coco_mapped/val.json',
        data_prefix=dict(img='srgb/images/val/'),
    )
)
test_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        ann_file='srgb/json_srgb_coco_mapped/test.json',
        data_prefix=dict(img='srgb/images/test/'),
    )
)

# ── Model ─────────────────────────────────────────────────────
# Standard DetDataPreprocessor — no RAW preprocessor module
# Detector is NOT frozen: this is inference / fine-tune upper bound
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=True,
        pad_size_divisor=32
    )
)

# Override the FreezeDetectorHook — detector must NOT be frozen for upper reference
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=10,
        rule='greater',
        min_delta=0.001
    ),
]

# ── WandB ─────────────────────────────────────────────────────
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='neuRAWns-mmdet-ROD',
            name=exp_name,
            config=dict(
                input='sRGB',
                preprocessor='None',
                detector='RTMDet-S',
                detector_frozen=False,
                note='Upper reference — full ISP sRGB input',
            )
        )
    )
]

visualizer = dict(vis_backends=vis_backends)
