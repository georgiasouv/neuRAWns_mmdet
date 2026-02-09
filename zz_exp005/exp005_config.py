exp_name = 'exp005'
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
# Load COCO pretrained Faster R-CNN instead of exp001
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# Dataset settings
dataset_type = 'CocoDataset'
data_root = '/cifs/Shares/Raw_Bayer_Datasets/ROD/'
# Map ROD classes to COCO classes
# ROD annotations have: Car(0), Cyclist(1), Pedestrian(2), Tram(3), Truck(4)
# We map to COCO: person(0), bicycle(1), car(2), train(6), truck(7)
classes = ('person', 'bicycle', 'car', 'train', 'truck')


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
        ann_file='json_raw_coco/train.json',
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
        ann_file='json_raw_coco/val.json',
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
        ann_file='json_raw_coco/test.json',
        data_prefix=dict(img='raw/test/'),
        pipeline=test_pipeline,
        metainfo=dict(classes=classes)))


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'json_raw_coco/val.json',
    metric='bbox',
    format_only=False,
    classwise=True)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'json_raw_coco/test.json',  
    metric='bbox',
    format_only=False,
    classwise=True)

# 
# HAS TO CHANGE
    
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
        experiment_name='exp004',
        save_raw=True,
        save_preprocessed=True
    )
]


vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
             'project': 'neuRAWns-mmdet-ROD',
             'name': 'exp004',
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