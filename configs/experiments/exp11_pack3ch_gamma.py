# ─────────────────────────────────────────────────────────────
# experiments/exp11_pack3ch_gamma.py
# Input:        Packed 3ch (averaged greens)
# Preprocessor: conv(x^(1/2.2)), fixed gamma, no gain
# Detector:     RTMDet-S / Frozen
# Dataset:      ROD
# ─────────────────────────────────────────────────────────────

_base_ = [
    '../_base_/detectors/rtmdetS_frozen.py',
    '../_base_/datasets/rod_dataset.py',
    '../_base_/default_runtime.py',
]

exp_name = 'exp11'

# ── Pipeline ──────────────────────────────────────────────────
train_pipeline = [
    dict(type='LoadRAWImageFromFile'),
    dict(type='NormaliseP99'),                                   # HDR → [0,1]
    dict(type='PackBayer', out_channels=3),                                  # [H/2, W/2, 3]
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadRAWImageFromFile'),
    dict(type='NormaliseP99'),
    dict(type='PackBayer', out_channels=3),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs')
]

# ── Override dataloader pipelines ─────────────────────────────
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader   = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader  = dict(dataset=dict(pipeline=test_pipeline))

# ── Model ─────────────────────────────────────────────────────
model = dict(
    data_preprocessor=dict(
        type='RAWDetDataPreprocessor',
        preprocessor_cfg=dict(
            type='ConvGamma',
            in_channels=3,
            out_channels=3,
            kernel_size=3,
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
            project='neuRAWns-mmdet-ROD',
            name=exp_name,
            config=dict(
                input='packed_3ch',
                preprocessor='ConvGamma',
                gamma=1/2.2,
                in_channels=3,
                kernel_size=3,
                gain=False,
                detector='RTMDet-S',
                detector_frozen=True,
            )
        )
    )
]

visualizer = dict(vis_backends=vis_backends)

# ── Hooks ─────────────────────────────────────────────────────
custom_hooks = [
    dict(
        type='FreezeDetectorHook',
        debug_mode=False,
        check_updates=False,
        priority='VERY_HIGH'
    ),
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=10,
        rule='greater',
        min_delta=0.001
    ),
    dict(
        type='SaveBatchImagesHook',
        save_dir='sample_images',
        experiment_name=exp_name,
        save_raw=True,
        save_preprocessed=True
    )
]


