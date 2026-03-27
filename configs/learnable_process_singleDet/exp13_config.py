exp_name = 'exp013'
DEBUG_MODE = True  

custom_imports = dict(
    imports=['modules.raw_preprocessors',
             'modules.raw_backbones',
             'modules.hooks',
             'datasets.pipelines',
             'mmengine.hooks'],
    allow_failed_imports=False
)

_base_ = [
    'mmdet::retinanet/retinanet_r50_fpn_1x_coco.py'
]

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'

dataset_type = 'CocoDataset'
data_root = '/cifs/Shares/WMGData/ROD/yolo/'
classes = ('person', 'bicycle', 'car', 'train', 'truck')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,            # RetinaNet standard is 0.01 (half of Faster R-CNN's 0.02)
        momentum=0.9,
        weight_decay=0.0001
    ),
    clip_grad=dict(max_norm=35, norm_type=2)
)

train_pipeline = [
    dict(type='LoadRAWImageFromFile'),
    dict(type='PackBayer'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadRAWImageFromFile'),
    dict(type='PackBayer'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs')
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=1
)

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='raw/json_raw_coco_mapped/train.json',
        data_prefix=dict(img='raw/images/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes)))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='raw/json_raw_coco_mapped/val.json',
        data_prefix=dict(img='raw/images/val/'),
        pipeline=test_pipeline,
        metainfo=dict(classes=classes)))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='raw/json_raw_coco_mapped/test.json',
        data_prefix=dict(img='raw/images/test/'),
        pipeline=test_pipeline,
        metainfo=dict(classes=classes)))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'raw/json_raw_coco_mapped/val.json',
    metric='bbox',
    format_only=False,
    classwise=True)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'raw/json_raw_coco_mapped/test.json',
    metric='bbox',
    format_only=False,
    classwise=True)

model = dict(
    type='RetinaNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        bgr_to_rgb=False,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='RAWResNet',           # ← identical to exp012, zero changes
        debug_mode=DEBUG_MODE,
        preprocess_cfg=dict(
            type='Exp012Processor',
            norm_threshold=0.99
        ),
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,              # RetinaNet FPN starts at P3 (level 1), not P2
        add_extra_convs='on_input',
        num_outs=5
    ),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,             # COCO head kept intact
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='FocalLoss',       # ← key difference from Faster R-CNN
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1
        ),
        sampler=dict(
            type='PseudoSampler'    # RetinaNet uses focal loss — no sampling needed
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100
    )
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=2,
        save_best='coco/bbox_mAP',
        rule='greater'
    )
)

custom_hooks = [
    dict(
        type='ClassMappingValidationHook',
        classes=classes,
        priority='VERY_HIGH'
    ),
    dict(
        type='FreezeDetectorHook',
        debug_mode=DEBUG_MODE,
        check_updates=DEBUG_MODE,
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
        experiment_name='exp013',
        save_raw=True,
        save_preprocessed=True
    )
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
             'project': 'neuRAWns-mmdet-ROD',
             'name': 'exp013',
             'config': {
                 'architecture': 'retinanet-r50',
                 'dataset': 'ROD',
                 'preprocessing': 'learning-based-frozen-detector'
             }
         })
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')