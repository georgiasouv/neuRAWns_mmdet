exp_name = 'exp005_sanity'

DEBUG_MODE = True

default_scope = 'mmdet' 

custom_imports = dict(
    imports=['mmdet.datasets',
             'mmdet.evaluation',
             'modules.raw_preprocessors',
             'modules.raw_backbones',
             'modules.hooks',
             'modules.multidetector_wrapper',
             'datasets.pipelines',
             'mmengine.hooks',
             'mmdet.visualization',
             'mmdet.models'],    
    allow_failed_imports=False
)

dataset_type = 'CocoDataset'
data_root = '/cifs/Shares/WMGData/ROD/'
classes = ('person', 'bicycle', 'car', 'train', 'truck')


model = dict(
    type='MultiDetectorModel',
    data_preprocessor=dict(         
        type='DetDataPreprocessor',
        mean=None,
        std=None,
        bgr_to_rgb=False,
        pad_size_divisor=32,
    ),
    preprocessor_cfg=dict(
        type='Exp002ConvBN'
    ),
    detector_cfgs=[
        dict(
            type='FasterRCNN',
            data_preprocessor=dict(
                type='DetDataPreprocessor',
                mean=None,
                std=None,
                bgr_to_rgb=False,
                pad_size_divisor=32
            ),
            backbone=dict(
                type='ResNet',
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=True,
                style='pytorch',
                init_cfg=dict(type='Pretrained', 
                              checkpoint='torchvision://resnet50')
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
                    target_means=[0., 0., 0., 0.],
                    target_stds=[1., 1., 1., 1.]
                ),
                loss_cls=dict(
                    type='CrossEntropyLoss', 
                    use_sigmoid=True, 
                    loss_weight=1.0
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
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.1, 0.1, 0.2, 0.2]
                    ),
                    loss_cls=dict(
                        type='CrossEntropyLoss', 
                        use_sigmoid=False, 
                        loss_weight=1.0
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
        ),
        dict(
            type='RTMDet',
            data_preprocessor=dict(
                type='DetDataPreprocessor',
                mean=None,
                std=None,
                bgr_to_rgb=False,
                pad_size_divisor=32
            ),
            backbone=dict(
                type='CSPNeXt',
                arch='P5',
                expand_ratio=0.5,
                deepen_factor=0.67,
                widen_factor=0.75,
                channel_attention=True,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='SiLU', inplace=True)
            ),
            neck=dict(
                type='CSPNeXtPAFPN',
                in_channels=[192, 384, 768],
                out_channels=192,
                num_csp_blocks=2,
                expand_ratio=0.5,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='SiLU', inplace=True)
            ),
            bbox_head=dict(
                type='RTMDetSepBNHead',
                num_classes=5,
                in_channels=192,
                stacked_convs=2,
                feat_channels=192,
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
                exp_on_reg=True,
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
        ),
        dict(
            type='DETR',
            data_preprocessor=dict(
                type='DetDataPreprocessor',
                mean=None,
                std=None,
                bgr_to_rgb=False,
                pad_size_divisor=1
            ),
            backbone=dict(
                type='ResNet',
                depth=50,
                num_stages=4,
                out_indices=(3,),
                frozen_stages=1,
                norm_cfg=dict(type='BN', requires_grad=False),
                norm_eval=True,
                style='pytorch',
                init_cfg=dict(type='Pretrained',
                            checkpoint='torchvision://resnet50')
            ),
            neck=dict(
                type='ChannelMapper',
                in_channels=[2048],
                kernel_size=1,
                out_channels=256,
                act_cfg=None,
                norm_cfg=None,
                num_outs=1
            ),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1
                    ),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=2048,
                        ffn_drop=0.1
                    )
                )
            ),
            decoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1
                    ),
                    cross_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1
                    ),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=2048,
                        ffn_drop=0.1
                    )
                ),
            ),
            positional_encoding=dict(
                num_feats=128,
                normalize=True
            ),
            bbox_head=dict(
                type='DETRHead',
                num_classes=5,
                embed_dims=256,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    bg_cls_weight=0.1,
                    use_sigmoid=False,
                    loss_weight=1.0,
                    class_weight=1.0
                ),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0)
            ),
            train_cfg=dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='ClassificationCost', weight=1.),
                        dict(type='BBoxL1Cost', weight=5., box_format='xywh'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.)
                    ]
                )
            ),
            test_cfg=dict(max_per_img=100)
        ),
        
    ],
    detector_ckpts=[
        'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth',
        'https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth',
    ],
    loss_weights=[1.0, 1.0, 1.0]
)


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

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

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

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='debug/annotations/coco_mapped_raw/train.json',
        data_prefix=dict(img='debug/raw/train/'),
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
        ann_file='debug/annotations/coco_mapped_raw/val.json',
        data_prefix=dict(img='debug/raw/val/'),
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
        ann_file='debug/annotations/coco_mapped_raw/test.json',
        data_prefix=dict(img='debug/raw/test/'),
        pipeline=test_pipeline,
        metainfo=dict(classes=classes)))


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'debug/annotations/coco_mapped_raw/val.json',
    metric='bbox',
    format_only=False,
    classwise=True)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'debug/annotations/coco_mapped_raw/test.json',
    metric='bbox',
    format_only=False,
    classwise=True)

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
        type='FreezeMultiDetectorHook',
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
        type='SaveBatchImagesHook_Multi',
        save_dir='sample_images',
        experiment_name=exp_name,
        save_raw=True,
        save_preprocessed=True
    )
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
             'project': 'neuRAWns-mmdet-ROD',
             'name': exp_name,
             'config': {
                 'architecture': 'multi-detector',
                 'detectors': 'faster-rcnn-r50+rtmdet-m+detr-r50',
                 'dataset': 'ROD',
                 'preprocessing': 'Exp005-residual-instancenorm'
             }
         })
]

visualizer = dict(
    type='Visualizer',
    vis_backends=vis_backends,
    name='visualizer')


