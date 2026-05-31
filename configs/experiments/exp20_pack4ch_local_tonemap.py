# ─────────────────────────────────────────────────────────────
# experiments/exp20_pack4ch_local_tonemap.py
# Input:        Packed 4ch (greens kept separate)
# Preprocessor: learnable curve + adaptive local contrast → 3×3 4→3
# Detector:     RTMDet-S / Frozen
# Dataset:      ROD
# ─────────────────────────────────────────────────────────────

_base_ = [
    '../_base_/detectors/rtmdetS_frozen.py',
    '../_base_/datasets/rod_dataset.py',
    '../_base_/default_runtime.py',
]

exp_name = 'exp20'

auto_scale_lr = dict(enable=True, base_batch_size=64)

train_pipeline = [
    dict(type='LoadRAWImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='NormaliseP99'),
    dict(type='PackBayer', out_channels=4),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadRAWImageFromFile'),
    dict(type='NormaliseP99'),
    dict(type='PackBayer', out_channels=4),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader   = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader  = dict(dataset=dict(pipeline=test_pipeline))

model = dict(
    data_preprocessor=dict(
        type='RAWDetDataPreprocessor',
        preprocessor_cfg=dict(
            type='GuidedLocalToneMap',
            in_channels=4,
            out_channels=3,
            kernel_size=3,
            knots=16,
            blur_k=15,
        ),
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=True,
        pad_size_divisor=32
    )
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='neuRAWns-mmdet-ROD-v2',
            name=exp_name,
            config=dict(
                input='packed_4ch',
                preprocessor='GuidedLocalToneMap',
                in_channels=4,
                kernel_size=3,
                knots=16,
                blur_k=15,
                learnable_curve=True,
                local_contrast=True,
                detector='RTMDet-S',
                detector_frozen=True,
                preprocessor_params=None,
                preprocessor_gmacs=None,
                preprocessor_latency_ms=None,
            )
        )
    )
]

visualizer = dict(vis_backends=vis_backends)

custom_hooks = [
    dict(type='FreezeDetectorHook', debug_mode=False,
         check_updates=False, priority='VERY_HIGH'),
    dict(type='EarlyStoppingHook', monitor='coco/bbox_mAP',
         patience=10, rule='greater', min_delta=0.001),
    dict(type='SaveBatchImagesHook', save_dir='sample_images',
         experiment_name=exp_name, save_raw=True, save_preprocessed=True),
    dict(type='PreprocessorMonitorHook', log_every_n_steps=50),
]