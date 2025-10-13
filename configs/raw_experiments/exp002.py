custom_imports = dict(
    imports=[
        'modules.raw_preprocessors',
        'modules.raw_backbones',
        'modules.hooks',
        'datasets.pipelines'
    ],
    allow_failed_imports=False
)

_base_ = [
    '../_base_/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# Optimizer - CRITICAL for your small trainable module
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',        # Good for small modules
        lr=0.001,            # Learning rate - START HERE, adjust if needed
        weight_decay=0.01
    ),
    clip_grad=dict(max_norm=35, norm_type=2)  # Gradient clipping
)

# Learning rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500           # Warmup iterations
    ),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        milestones=[8, 11],  # Drop LR at these epochs
        gamma=0.1
    )
]

# Training schedule
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,           # Total epochs - adjust based on convergence
    val_interval=1           # Validate every epoch
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,                    # Save every epoch
        max_keep_ckpts=3,              # Keep last 3 (in case of crashes)
        save_best='auto',              # Save best model
        rule='greater'                 # Higher mAP = better (default, but explicit)
    ),
    logger=dict(
        type='LoggerHook',
        interval=50                    # Log every 50 iterations
    )
)
dataset_type = 'CocoDataset'
data_root = '/cifs/Shares/Raw_Bayer_Datasets/ROD/'
classes = ('Car', 'Cyclist', 'Pedestrian', 'Tram', 'Truck')

train_pipeline = [
    dict(type='LoadRAWImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadRAWImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='json_raw/train.json',
        data_prefix=dict(img='raw/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes)))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='json_raw/val.json',
        data_prefix=dict(img='raw/val/'),
        pipeline=test_pipeline,
        metainfo=dict(classes=classes)))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='json_raw/test.json',  # Different annotation file
        data_prefix=dict(img='raw/test/'),  # Different image folder
        pipeline=test_pipeline,
        metainfo=dict(classes=classes)))


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'json_raw/val.json',
    metric='bbox',
    format_only=False)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'json_raw/test.json',  
    metric='bbox',
    format_only=False)

model = dict(
    type='FasterRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=None,      
        std=None,       # No std normalisation
        bgr_to_rgb=False,          
        pad_size_divisor=32        
    ),
    backbone=dict(
        type='RAWResNet',  
        preprocess_cfg=dict(
            type='Exp002ConvBN',
            in_channels=4,
            out_channels=3,
            norm_threshold=0.99
        ),
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),  # Output C2, C3, C4, C5
        frozen_stages=1,            # Freeze conv1 + C2 (first 2 stages)
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,             # Keep BN in eval mode
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  # ResNet-50 output channels
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=5,  # Your 5 classes
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        )
    ),
    
    # Training config
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True
            ),
            pos_weight=-1,
            debug=False
        )
    ),
    
    # Test config
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100
        )
    )
)

custom_hooks = [
    dict(type='FreezeDetectorHook', priority='VERY_HIGH')
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
             'project': 'rod-raw-preprocessing',
             'name': 'exp002',
             'config': {
                 'architecture': 'faster-rcnn-r50',
                 'dataset': 'ROD',
                 'preprocessing': 'raw-learnable'
             }
         })
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')