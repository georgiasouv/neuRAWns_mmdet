# ─────────────────────────────────────────────────────────────
# _base_/datasets/rod_dataset.py
# ROD dataset — paths, classes, evaluators.
# Does NOT define pipelines — each experiment sets its own.
# ─────────────────────────────────────────────────────────────

dataset_type = 'CocoDataset'
data_root = '/home/souval_g_WMGDS.WMG.WARWICK.AC.UK/data/ROD/yolo/'

# ROD classes mapped to COCO IDs:
# Car→car(2), Cyclist→bicycle(1), Pedestrian→person(0), Tram→train(6), Truck→truck(7)
classes = ('person', 'bicycle', 'car', 'train', 'truck')

# ── Dataloaders ───────────────────────────────────────────────
# Pipelines are intentionally omitted here.
# Each experiment config defines train_pipeline / test_pipeline
# and passes them via:
#   train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='raw/json_raw_coco_mapped/train.json',
        data_prefix=dict(img='raw/images/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=classes),
        pipeline=None  # overridden per experiment
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='raw/json_raw_coco_mapped/val.json',
        data_prefix=dict(img='raw/images/val/'),
        metainfo=dict(classes=classes),
        pipeline=None  # overridden per experiment
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='raw/json_raw_coco_mapped/test.json',
        data_prefix=dict(img='raw/images/test/'),
        metainfo=dict(classes=classes),
        pipeline=None  # overridden per experiment
    )
)

# ── Evaluators ────────────────────────────────────────────────
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'raw/json_raw_coco_mapped/val.json',
    metric='bbox',
    format_only=False,
    classwise=True
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'raw/json_raw_coco_mapped/test.json',
    metric='bbox',
    format_only=False,
    classwise=True
)