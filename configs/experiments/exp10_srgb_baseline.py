# ─────────────────────────────────────────────────────────────
# experiments/exp10_srgb_baseline_test.py
# Input:        sRGB images (standard JPEG)
# Preprocessor: None — detector receives sRGB directly
# Detector:     RTMDet-S / COCO pretrained, frozen
# Dataset:      ROD (sRGB split)
# Purpose:      Upper reference — best case for the frozen detector
# ─────────────────────────────────────────────────────────────

_base_ = [
    '../_base_/detectors/rtmdetS_frozen.py',
    '../_base_/default_runtime.py',
]

exp_name = 'exp10'

# ── Pipeline ──────────────────────────────────────────────────
# Standard sRGB pipeline — no RAW loading, no packing, no P99 norm
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs')
]

# ── Dataloaders ───────────────────────────────────────────────
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='/networkhome/WMGDS/souval_g/data/ROD/yolo/',
        ann_file='srgb/json_raw_coco_mapped/test.json',
        data_prefix=dict(img='srgb/images/test/'),
        pipeline=test_pipeline
    )
)

# ── Evaluator ─────────────────────────────────────────────────
test_evaluator = dict(
    type='CocoMetric',
    ann_file='/networkhome/WMGDS/souval_g/data/ROD/yolo/srgb/json_raw_coco_mapped/test.json',
    metric='bbox',
    classwise=True,
    label_to_catid={0: 1, 1: 2, 2: 3, 6: 7, 7: 8}
)

# ── Model override ────────────────────────────────────────────
# Use standard DetDataPreprocessor — no RAWDetDataPreprocessor needed
model = dict(
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None
    )
)

# ── Hooks ─────────────────────────────────────────────────────
# No FreezeDetectorHook needed — no training, no preprocessor to freeze
# No EarlyStoppingHook — test only
custom_hooks = []

train_cfg = None
optim_wrapper = None
param_scheduler = None
train_dataloader = None
val_cfg = None
val_dataloader = None
val_evaluator = None
# ── WandB ─────────────────────────────────────────────────────
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='neuRAWns-mmdet-ROD-v2',
            name=exp_name,
            config=dict(
                input='sRGB',
                preprocessor='None',
                detector='RTMDet-S',
                detector_frozen=True,
                purpose='upper_reference',
            )
        )
    )
]

visualizer = dict(vis_backends=vis_backends)