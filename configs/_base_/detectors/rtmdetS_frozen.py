# ─────────────────────────────────────────────────────────────
# _base_/detectors/rtmdetS_frozen.py
# RTMDet-S with COCO pretrained weights, detector frozen.
# Preprocessor module is NOT defined here — set per experiment.
# ─────────────────────────────────────────────────────────────

_base_ = ['mmdet::rtmdet/rtmdet_s_8xb32-300e_coco.py']

# COCO pretrained RTMDet-S weights
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387f0a22.pth'

# ── Model overrides ───────────────────────────────────────────
# Override num_classes from 80 (COCO) to 5 (ROD mapped classes).
# data_preprocessor is overridden per experiment because
# the RAWDetDataPreprocessor carries the learnable module.
model = dict(
    bbox_head=dict(num_classes=5)
)

# ── Optimiser ─────────────────────────────────────────────────
# Only the preprocessor module trains — detector is frozen.
# Small LR appropriate for a ~few-hundred-param module.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.01)
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=100
    )
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=1
)

# ── Hooks ─────────────────────────────────────────────────────
# FreezeDetectorHook and EarlyStoppingHook are shared by all
# frozen-detector experiments.
# SaveBatchImagesHook and WandB name are set per experiment.
custom_hooks = [
    dict(
        type='FreezeDetectorHook',
        debug_mode=False,         # overridden to True in experiment if needed
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
]