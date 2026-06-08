# ─────────────────────────────────────────────────────────────
# configs/deploy/deploy_TEMPLATE.py
# Deployment / LODO-evaluation config: trained tone module + ONE
# detector. Copy per detector (deploy_retinanet.py, deploy_detr.py,
# deploy_faster_rcnn.py, plus the held-out one for LODO).
#
# THE INVARIANT THAT MAKES PARITY HOLD:
#   data_preprocessor and preprocessor below must be copied VERBATIM
#   from the training config. Any difference (linear vs log norm,
#   different black/white levels, different module variant or knot
#   spacing) silently changes the input distribution and parity
#   breaks. tools/parity_check.py exists to catch exactly this.
# ─────────────────────────────────────────────────────────────

custom_imports = dict(
    imports=[
        'modules.raw_preprocessors',                 # NormaliseLinear/Log + tone modules
        'modules.wrappers.single_detector_deploy',
    ],
    allow_failed_imports=False)

model = dict(
    type='SingleDetectorDeploy',

    # ── COPY VERBATIM from training config ──────────────────
    data_preprocessor=dict(
        # e.g. your RAW packing + fixed normalisation stage:
        # type='RAWDetDataPreprocessor',
        # normalise=dict(type='NormaliseLinear',
        #                black_level=8.5, white_level=4015028),
        # ...
    ),
    preprocessor=dict(
        # e.g. type='LocalCurve', grid=(8, 8), thumbnail=64,
        #      knot_spacing='log', out_scale=255, ...
    ),
    # ────────────────────────────────────────────────────────

    # The ONE detector for this deployment (its full mmdet model cfg,
    # typically pulled in via _base_):
    detector=dict(
        # _base_-merged detector cfg here
    ),

    # Published COCO weights for the detector (for LODO, this is the
    # held-out detector the preprocessor never trained against):
    detector_checkpoint='checkpoints/<detector>_coco.pth',

    # The trained ensemble checkpoint; ONLY preprocessor.* keys are
    # extracted (strict load, verified):
    ensemble_checkpoint='work_dirs/<exp>/epoch_<N>.pth',
)

# Reuse the training config's val_dataloader / val_evaluator via
# _base_ so evaluation runs on identical data + metrics.