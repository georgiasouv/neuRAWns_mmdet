# ─────────────────────────────────────────────────────────────
# _base_/default_runtime.py
# Logging, checkpointing, custom imports.
# WandB run `name` is overridden per experiment.
# ─────────────────────────────────────────────────────────────

custom_imports = dict(
    imports=[
        'modules.raw_preprocessors',   # ConvGamma3ch, ConvGamma4ch, etc.
        'modules.wrappers',
        'modules.hooks',               # FreezeDetectorHook, SaveBatchImagesHook, etc.
        'datasets.pipelines',          # PackBayer_3ch, PackBayer_4ch, LoadRAWImageFromFile
    ],
    allow_failed_imports=False
)

# ── Runtime ───────────────────────────────────────────────────
default_scope = 'mmdet'

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
resume = False

# ── Default hooks ─────────────────────────────────────────────
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=2,
        save_best='coco/bbox_mAP',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

vis_backends = [
    dict(type='LocalVisBackend'),
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)















# default_scope = 'mmdet'

# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),
#     logger=dict(type='LoggerHook', interval=50),
#     param_scheduler=dict(type='ParamSchedulerHook'),
#     checkpoint=dict(type='CheckpointHook', interval=1),
#     sampler_seed=dict(type='DistSamplerSeedHook'),
#     visualization=dict(type='DetVisualizationHook'))

# env_cfg = dict(
#     cudnn_benchmark=False,
#     mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
#     dist_cfg=dict(backend='nccl'),
# )

# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
# log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

# log_level = 'INFO'
# load_from = None
# resume = False
