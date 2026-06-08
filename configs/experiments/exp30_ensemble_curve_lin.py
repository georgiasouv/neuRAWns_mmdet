# ─────────────────────────────────────────────────────────────
# experiments/exp30_ensemble_curve_lin.py
# Ensemble version of the validated exp22 recipe, matched to the
# corrected MultiDetectorModel (list-based interface).
#
#   Module:        LocalCurve  (SHARED, top-level, lifted out of
#                  RAWDetDataPreprocessor)
#   Normalisation: NormaliseLinear  (in the pipeline, unchanged)
#   Detectors:     RTMDet-S (one-stage) + Faster R-CNN (two-stage)
#                  + DETR (set-prediction), ALL frozen
#   Dataset:       ROD
# ─────────────────────────────────────────────────────────────
_base_ = [
    '../_base_/datasets/rod_dataset.py',
    '../_base_/default_runtime.py',
    # Each file must expose a UNIQUELY-NAMED full model dict that
    # INCLUDES its own data_preprocessor (mean/std/bgr_to_rgb) and the
    # 'type' key (RTMDet / FasterRCNN / DETR), e.g. inside the file:
    #   rtmdet_model = dict(type='RTMDet', data_preprocessor=dict(...), ...)
    '../_base_/detectors/rtmdetS_model.py',        # -> rtmdet_model
    '../_base_/detectors/faster_rcnn_model.py',    # -> faster_rcnn_model
    '../_base_/detectors/detr_model.py',           # -> detr_model
]

exp_name = 'exp30_ensemble_curve_lin'
auto_scale_lr = dict(enable=True, base_batch_size=64)

custom_imports = dict(
    imports=[
        'datasets.pipelines.loading',
        'datasets.pipelines.normalise_raw',
        'modules.raw_preprocessors',
        'modules.wrappers.multi_detector_model',   # MultiDetectorModel (corrected)
        'modules.loops.ensemble_val_loop',         # EnsembleValLoop
        'modules.hooks.ensemble_diagnostics',      # freeze / grad / enhanced-stats
        'modules.hooks.ensemble_guards',           # nan / update / det-input-contract
        'modules.hooks.preproc_observability',     # tone-params / norm-input / identity-init
        # Add ONLY when you switch train_cfg to EnsembleTrainLoop below:
        # 'modules.loops.ensemble_train_loop',
    ],
    allow_failed_imports=False)



# ── Pipeline (IDENTICAL to exp22 — shared across all detectors) ──
norm = dict(type='NormaliseLinear', black_level=8.5, white_level=4015028.0)
train_pipeline = [
    dict(type='LoadRAWImageFromFile'),
    norm,
    dict(type='PackBayer', out_channels=4),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadRAWImageFromFile'),
    norm,
    dict(type='PackBayer', out_channels=4),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# ── Shared tone module (lifted VERBATIM from exp22.preprocessor_cfg) ──
preprocessor = dict(
    type='LocalCurve',
    in_channels=4,
    out_channels=3,
    grid_h=8,
    grid_w=8,
    thumb_size=64,
    hidden=32,
    knot_spacing='log',
    n_knots=16,
    out_scale=255.0,
)

# ── Model ─────────────────────────────────────────────────────
model = dict(
    type='MultiDetectorModel',
    # Top-level: PACK + PAD ONLY. No COCO mean/std here — each detector
    # owns its own normalisation. preprocessor_cfg=None means this
    # container no longer runs the tone module.
    data_preprocessor=dict(
        type='RAWDetDataPreprocessor',
        preprocessor_cfg=None,
        mean=None,
        std=None,
        bgr_to_rgb=False,
        pad_size_divisor=32,
    ),
    preprocessor_cfg=preprocessor,                 # the SHARED learnable module
    detector_cfgs=[                                # each = full model dict incl. 'type' + its data_preprocessor
        {{_base_.rtmdet_model}},
        {{_base_.faster_rcnn_model}},
        {{_base_.detr_model}},
    ],
    detector_ckpts=[
    'checkpoints/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth',
    'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
    'checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth',
    ],
    # loss_weights=[1.0, 1.0, 1.0],                # default = all 1.0 (naive sum)
)

# ── Loops ─────────────────────────────────────────────────────
# Floor: naive-sum training via the default loop (mmengine sums all
# loss tensors the model returns). This is the clean first test of the
# core idea. Per-detector GRADIENT diagnostics (grad/norm_*, grad/cos_*)
# stay idle here because only EnsembleTrainLoop stashes
# _last_per_detector_grads — every OTHER diagnostic still fires.
# To enable them later: uncomment the import above and swap the line
# below to: train_cfg = dict(type='EnsembleTrainLoop', max_epochs=50, val_interval=1)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='EnsembleValLoop')             # per-detector mAP vector
test_cfg = dict(type='EnsembleValLoop')

# Only the shared preprocessor trains; detectors are frozen in the
# wrapper. A standard optimizer is fine — frozen params have no grad.
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=1e-4))

# ── Hooks (exp22 set reconciled with the ensemble diagnostics) ──
custom_hooks = [
    # Verifies each detector in the ModuleList stays frozen (the
    # train() override should make this a no-op, but it's the proof).
    dict(type='VerifyMultiDetectorFreezeHook', priority='VERY_HIGH'),
    dict(type='NaNGuardHook', priority='VERY_HIGH'),
    dict(type='GradDiagnosticHook', log_interval=50),      # idle until EnsembleTrainLoop
    dict(type='EnhancedStatsHook', log_interval=50),
    dict(type='DetectorInputContractHook', log_interval=50),
    dict(type='ToneParamDiagnosticHook', log_interval=50),
    dict(type='NormInputStatsHook', log_interval=50),
    dict(type='IdentityInitProbeHook'),
    dict(type='PreprocessorUpdateHook', log_interval=100),
    dict(type='SaveBatchImagesHook', save_dir='sample_images',
         experiment_name=exp_name, save_raw=True, save_preprocessed=True),
    # CHANGED MONITOR KEY: under EnsembleValLoop 'coco/bbox_mAP' no
    # longer exists — metrics are detector-prefixed.
    dict(type='EarlyStoppingHook', monitor='ensemble/mean_mAP',
         patience=10, rule='greater', min_delta=0.001),
]

# default_runtime's CheckpointHook save_best must use the prefixed key.
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1,
                    save_best='ensemble/mean_mAP', rule='greater'))

# ── WandB ─────────────────────────────────────────────────────
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend', init_kwargs=dict(
        project='neuRAWns-mmdet-ROD-v2',
        name=exp_name,
        config=dict(
            phase=3,
            input='packed_4ch',
            preprocessor='LocalCurve',
            normalisation='NormaliseLinear',
            knot_spacing='log',
            grid='8x8',
            out_scale=255.0,
            detectors=['RTMDet-S', 'Faster-RCNN-R50', 'DETR-R50'],
            detectors_frozen=True,
            supervision='ensemble-naive-sum',
        ))),
]
visualizer = dict(vis_backends=vis_backends)