# transforms/normalise_raw.py
#
# Fixed (content-independent) RAW normalisations — the Option-1 / Option-2
# pair for the normalisation x module experiments.  Both replace NormaliseP99
# in the pipeline:  LoadRAWImageFromFile -> Normalise{Linear,Log} -> PackBayer
#
# Why fixed rather than per-image P99:
#   Per-image P99 adaptively pre-exposes every frame OUTSIDE the learnable
#   module (a night frame gets divided by its brightest headlight), which
#   pre-empts the adaptivity the module is supposed to learn and makes the
#   normalisation content-dependent. These two are the same transform for
#   every image; all adaptivity lives in learned weights.
#
# Sensor constants (black/white level) are per-camera metadata supplied via
# the dataset config — the transform itself stays generic across sensors.

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class NormaliseLinear(BaseTransform):
    """Option 1 — fixed-scale LINEAR normalisation.

        y = clip((x - black_level) / (white_level - black_level), 0, 1)

    Preserves true relative HDR brightness: a night pedestrian stays ~5e-5,
    a day scene stays ~1e-2..1.  The module receives the raw dynamic range
    and must do ALL tone compression itself (log-spaced knots essential).

    Saturated pixels (the sentinel, e.g. 15,990,553 on ROD's IMX490) sit
    above white_level and clip to exactly 1.0 — physically "saturated".

    Args:
        black_level (float): sensor black level (ROD: 8.5)
        white_level (float): sensor white level (ROD: 4_015_028.0)
    """

    def __init__(self, black_level: float, white_level: float):
        assert white_level > black_level
        self.black_level = float(black_level)
        self.white_level = float(white_level)
        self._scale = 1.0 / (self.white_level - self.black_level)

    def transform(self, results: dict) -> dict:
        img = results['img'].astype(np.float32)
        img = (img - self.black_level) * self._scale
        results['img'] = np.clip(img, 0.0, 1.0)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(black_level={self.black_level}, '
                f'white_level={self.white_level})')


@TRANSFORMS.register_module()
class NormaliseLog(BaseTransform):
    """Option 2 — fixed LOG normalisation.

        y = log1p(max(x - black_level, 0)) / log1p(white_level - black_level)

    Front-loads the HDR compression into the (non-learnable) pipeline:
    the ~20-stop linear range maps to [0,1] with resolution allocated
    per-stop rather than per-linear-unit, so night signal (~173 counts)
    lands around y~0.34 instead of y~4e-5.  The learnable module then
    differentiates on SPATIAL capacity, not tonal-resolution rescue.
    Still content-independent: identical transform for every image.

    Args:
        black_level (float): sensor black level (ROD: 8.5)
        white_level (float): sensor white level (ROD: 4_015_028.0)
    """

    def __init__(self, black_level: float, white_level: float):
        assert white_level > black_level
        self.black_level = float(black_level)
        self.white_level = float(white_level)
        self._denom = float(np.log1p(self.white_level - self.black_level))

    def transform(self, results: dict) -> dict:
        img = results['img'].astype(np.float32)
        img = np.maximum(img - self.black_level, 0.0)
        img = np.log1p(img) / self._denom
        results['img'] = np.clip(img, 0.0, 1.0)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(black_level={self.black_level}, '
                f'white_level={self.white_level})')