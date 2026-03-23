# import os.path as osp

# _config_name = osp.splitext(osp.basename(__file__))[0]
# work_dir = f'work_dirs/{_config_name}'

custom_imports = dict(
    imports=['mmengine.hooks'],
    allow_failed_imports=False
)

_base_ = [
     'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
]

# Dataset settings
dataset_type = 'CocoDataset'
data_root = '/cifs/Shares/Raw_Bayer_Datasets/ROD/'
classes = ('Car', 'Cyclist', 'Pedestrian', 'Tram', 'Truck')


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.02,  # ← ADD THIS
        momentum=0.9,
        weight_decay=0.0001
    ),
    clip_grad=dict(max_norm=35, norm_type=2)  
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
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
    dict(type='LoadImageFromFile'),
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
        ann_file='json_isp/train.json',
        data_prefix=dict(img='isp/train/'),
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
        ann_file='json_isp/val.json',
        data_prefix=dict(img='isp/val/'),
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
        ann_file='json_isp/test.json',  # Different annotation file
        data_prefix=dict(img='isp/test/'),  # Different image folder
        pipeline=test_pipeline,
        metainfo=dict(classes=classes)))


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'json_isp/val.json',
    metric='bbox',
    format_only=False)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'json_isp/test.json',  # Match test annotations
    metric='bbox',
    format_only=False)

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=5)))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,                    # Save every epoch
        max_keep_ckpts=2,             # Keep only last 3 checkpoints
        save_best='coco/bbox_mAP',    # Also save best mAP checkpoint
        rule='greater'                 # Higher mAP is better
    )
    
)

# custom_hooks is a list because you're adding new hooks that don't exist in the defaults
#
custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=10,
        rule='greater',
        min_delta=0.001
    )
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
             'project': 'neuRAWns-mmdet-ROD',
             'name': 'exp001-baseline-faster-rcnn-r50',
             'config': {
                 'architecture': 'faster-rcnn-r50',
                 'dataset': 'ROD',
                 'preprocessing': 'traditional-isp'
             }
         })
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

