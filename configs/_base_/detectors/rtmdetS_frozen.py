# ─────────────────────────────────────────────────────────────
# _base_/detectors/rtmdet_s_frozen.py
# RTMDet-S model definition only — no dataset, no runtime.
# Inheriting from the full mmdet RTMDet config is intentionally
# avoided because it pulls in COCO dataset keys that conflict
# with rod_dataset.py when both are used as bases.
# ─────────────────────────────────────────────────────────────

# COCO pretrained RTMDet-S weights
load_from = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/.cache/torch/hub/checkpoints/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'

# ── Model ─────────────────────────────────────────────────────
# Full RTMDet-S model definition.
# data_preprocessor is intentionally omitted — defined per experiment
# because the RAWDetDataPreprocessor carries the learnable module.
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None
    ),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)
    ),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=5,           # ROD mapped classes, not 80
        in_channels=128,
        stacked_convs=2,
        feat_channels=128,
        anchor_generator=dict(
            type='MlvlPointGenerator',
            offset=0,
            strides=[8, 16, 32]
        ),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=False,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)
    ),
    train_cfg=dict(
        assigner=dict(
            type='DynamicSoftLabelAssigner',
            topk=13
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300
    )
)

# ── Optimiser ─────────────────────────────────────────────────
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

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ── Shared custom hooks ────────────────────────────────────────
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
]

# wget -P ~/.cache/torch/hub/checkpoints \
#   https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth