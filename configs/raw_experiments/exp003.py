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
     'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
]

# Load pretrained detector from exp001 (RGB baseline)
load_from = '/networkhome/WMGDS/souval_g/neuRAWns_mmdet/work_dirs/exp001_faster_rcnn_r50_fpn/best_coco_bbox_mAP_epoch_19.pth'

# Dataset settings
dataset_type = 'CocoDataset'
data_root = '/cifs/Shares/Raw_Bayer_Datasets/ROD/'
classes = ('Car', 'Cyclist', 'Pedestrian', 'Tram', 'Truck')


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.02, 
        momentum=0.9,
        weight_decay=0.0001
    ),
    clip_grad=dict(max_norm=35, norm_type=2)  
)

train_pipeline = [
    dict(type='LoadRAWImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,  
    val_interval=1
)

test_pipeline = [
    dict(type='LoadRAWImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=1,
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
    num_workers=1,
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
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='json_raw/test.json',
        data_prefix=dict(img='raw/test/'),
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
        std=None,    
        bgr_to_rgb=False,          
        pad_size_divisor=32        
    ),
    backbone=dict(
        type='RAWResNet',
        debug_mode=DEBUG_MODE,  
        preprocess_cfg=dict(
            type='Exp002ConvBN',
            in_channels=4,
            out_channels=3,
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
            num_classes=5,
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
        experiment_name='exp003',
        save_raw=True,
        save_preprocessed=True
    )
]


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
             'project': 'neuRAWns-mmdet-ROD',
             'name': 'exp003-frozen-detector',
             'config': {
                 'architecture': 'faster-rcnn-r50',
                 'dataset': 'ROD',
                 'preprocessing': 'learning-based-frozen-detector'
             }
         })
]


visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')