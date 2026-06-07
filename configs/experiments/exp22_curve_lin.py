# ─────────────────────────────────────────────────────────────
# experiments/exp22_curve_lin.py   [AUTO-GENERATED Phase-2 grid]
# Module:        LocalCurve
# Normalisation: NormaliseLinear  (Option-1 fixed-linear: module sees raw HDR range)
# Detector:      RTMDet-S / Frozen     Dataset: ROD
# Grid cell:     Phase-2 normalisation x spatial-module experiment
# ─────────────────────────────────────────────────────────────

_base_ = [
    '../_base_/detectors/rtmdetS_frozen.py',
    '../_base_/datasets/rod_dataset.py',
    '../_base_/default_runtime.py',
]

exp_name = 'exp22_curve_lin'

auto_scale_lr = dict(enable=True, base_batch_size=64)

# ── Pipeline ──────────────────────────────────────────────────
# ROD / Sony IMX490 sensor constants (from dataset profiling)
norm = dict(type='NormaliseLinear', black_level=8.5, white_level=4015028.0)

train_pipeline = [
    dict(type='LoadRAWImageFromFile'),
    norm,
    dict(type='PackBayer', out_channels=4),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadRAWImageFromFile'),
    norm,
    dict(type='PackBayer', out_channels=4),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader   = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader  = dict(dataset=dict(pipeline=test_pipeline))

# ── Model ─────────────────────────────────────────────────────
model = dict(
    data_preprocessor=dict(
        type='RAWDetDataPreprocessor',
        preprocessor_cfg=dict(
            type='LocalCurve',
            in_channels=4,
            out_channels=3,
            grid_h=8,
            grid_w=8,
            thumb_size=64,
            hidden=32,
            knot_spacing='log',     # held constant in Phase 2 (Phase-3 ablation)
            n_knots=16,
            out_scale=255.0,        # COCO-scale output for frozen detector mean/std
        ),
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    )
)

# ── WandB ─────────────────────────────────────────────────────
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(
        project='neuRAWns-mmdet-ROD-v2',
        name=exp_name,
        config=dict(
            phase=2,
            input='packed_4ch',
            preprocessor='LocalCurve',
            normalisation='NormaliseLinear',
            knot_spacing='log',
            grid='8x8',
            out_scale=255.0,
            detector='RTMDet-S',
            detector_frozen=True,
        ))),
]
visualizer = dict(vis_backends=vis_backends)

# ── Hooks ─────────────────────────────────────────────────────
custom_hooks = [
    dict(type='FreezeDetectorHook', debug_mode=False, check_updates=False, priority='VERY_HIGH'),
    dict(type='EarlyStoppingHook', monitor='coco/bbox_mAP', patience=10, rule='greater', min_delta=0.001),
    dict(type='SaveBatchImagesHook', save_dir='sample_images', experiment_name=exp_name,
         save_raw=True, save_preprocessed=True),
    dict(type='PreprocessorMonitorHook', log_every_n_steps=50),
]