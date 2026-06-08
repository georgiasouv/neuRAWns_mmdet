# ─────────────────────────────────────────────────────────────
# modules/wrappers/single_detector_deploy.py
# The INFERENCE-TIME artifact: frozen trained tone module + ONE
# detector. This is what "deployment" and the LODO evaluation
# actually run. It mirrors MultiDetectorWrapper's Option-A path
# exactly:
#
#   fixed-normalised RAW (top-level data_preprocessor, must be
#   IDENTICAL to the training config's)
#     -> tone module -> [0,255]
#     -> THIS detector's own DetDataPreprocessor (COCO mean/std)
#     -> detector.predict
#
# Checkpoint logic:
#   - detector_checkpoint : the detector's published COCO weights
#     (LODO held-out case) or whatever weights it had during training
#   - ensemble_checkpoint : a MultiDetectorWrapper checkpoint; only
#     the 'preprocessor.*' keys are extracted, loaded STRICTLY, and
#     the load is verified. A wrong/empty extraction raises.
#
# Everything is frozen + eval() — this class cannot train.
# ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.runner.checkpoint import _load_checkpoint, load_checkpoint
from mmdet.registry import MODELS


def extract_preprocessor_state(ensemble_ckpt_path, prefix='preprocessor.'):
    """Pull the tone module's state_dict out of an ensemble checkpoint.

    Returns the sub-state-dict with the prefix stripped. Raises if no
    keys match — that means the checkpoint isn't a MultiDetectorWrapper
    checkpoint or the prefix changed.
    """
    ckpt = _load_checkpoint(ensemble_ckpt_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt)
    sub = {k[len(prefix):]: v for k, v in state.items()
           if k.startswith(prefix)}
    if not sub:
        raise RuntimeError(
            f'[deploy] no keys with prefix "{prefix}" found in '
            f'{ensemble_ckpt_path}. Keys look like: '
            f'{list(state.keys())[:5]} ...')
    return sub


@MODELS.register_module()
class SingleDetectorDeploy(BaseModel):

    def __init__(self,
                 preprocessor,
                 detector,
                 detector_checkpoint=None,
                 ensemble_checkpoint=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)

        self.preprocessor = MODELS.build(preprocessor)
        self.detector = MODELS.build(detector)

        if detector_checkpoint is not None:
            self._load_and_verify_detector(detector_checkpoint)

        if ensemble_checkpoint is not None:
            sub = extract_preprocessor_state(ensemble_checkpoint)
            # strict=True: a single missing/unexpected key raises.
            # If this fires, the deploy preprocessor cfg does not match
            # the trained module's architecture (e.g. different variant
            # or knot count) — fix the config, don't relax strictness.
            self.preprocessor.load_state_dict(sub, strict=True)
            print(f'[deploy] preprocessor: loaded {len(sub)} tensors '
                  f'from {ensemble_checkpoint} (strict) OK')

        # This artifact never trains.
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def _load_and_verify_detector(self, path):
        ckpt = _load_checkpoint(path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        model_keys = set(self.detector.state_dict().keys())
        coverage = len(model_keys & set(state.keys())) / max(len(model_keys), 1)
        load_checkpoint(self.detector, path, map_location='cpu',
                        strict=False)
        if coverage < 0.99:
            raise RuntimeError(
                f'[deploy] detector checkpoint {path} covers only '
                f'{coverage:.1%} of the model keys — wrong checkpoint '
                f'or wrong detector config (the strict=False trap).')
        print(f'[deploy] detector: checkpoint key coverage '
              f'{coverage:.1%} OK')

    def train(self, mode=True):
        # Defensive: even if something calls .train(), stay in eval.
        return super().train(False)

    def _enhance(self, batch_inputs):
        return self.preprocessor(batch_inputs)

    def predict(self, batch_inputs, batch_data_samples, rescale=True):
        enhanced = self._enhance(batch_inputs)
        data = {'inputs': enhanced, 'data_samples': batch_data_samples}
        data = self.detector.data_preprocessor(data, training=False)
        return self.detector.predict(data['inputs'], data['data_samples'],
                                     rescale=rescale)

    def forward(self, inputs, data_samples=None, mode='predict'):
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        raise ValueError(
            f'SingleDetectorDeploy is inference-only; got mode={mode!r}')