exp_name = 'exp014'
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
    'mmdet::dino/dino-4scale_r50_improved_8xb2-12e_coco.py'
]

load_from = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/.cache/torch/hub/checkpoints/dino-4scale_r50_improved_8xb2-12e_coco_20230818_162607-6f47a913.pth'

dataset_type = 'CocoDataset'
data_root = '/cifs/Shares/WMGData/ROD/yolo/'
classes = ('person', 'bicycle', 'car', 'train', 'truck')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=1e-4
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2)  # DINO needs aggressive grad clipping
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
    type='DINO',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        bgr_to_rgb=False,
        pad_size_divisor=1          # transformers don't need stride-aligned padding
    ),
    backbone=dict(
        type='RAWResNet',           # ← same wrapper, but out_indices is different
        debug_mode=DEBUG_MODE,
        preprocess_cfg=dict(
            type='Exp012Processor',
            norm_threshold=0.99
        ),
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),      # ← P3/P4/P5 only — DINO drops the finest scale
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='ChannelMapper',       # ← replaces FPN entirely
        in_channels=[512, 1024, 2048],  # matches out_indices=(1,2,3)
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4                  # adds one extra level via strided conv
    ),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_levels=4,       # must match neck num_outs
                dropout=0.0
            ),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0
            )
        )
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,   # needed for auxiliary losses during training
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                dropout=0.0
            ),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_levels=4,
                dropout=0.0
            ),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0
            )
        ),
        post_norm_cfg=None
    ),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20
    ),
    bbox_head=dict(
        type='DINOHead',
        num_classes=80,             # COCO head kept intact
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)
    ),
    dn_cfg=dict(                    # denoising training — DINO's key improvement over DETR
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)
    ),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ]
        )
    ),
    test_cfg=dict(max_per_img=300)  # transformers generate many candidates
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
        experiment_name='exp014',
        save_raw=True,
        save_preprocessed=True
    )
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
             'project': 'neuRAWns-mmdet-ROD',
             'name': 'exp014',
             'config': {
                 'architecture': 'dino-4scale-r50',
                 'dataset': 'ROD',
                 'preprocessing': 'learning-based-frozen-detector'
             }
         })
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')