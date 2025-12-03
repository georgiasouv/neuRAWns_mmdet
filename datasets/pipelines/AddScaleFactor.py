from mmdet.registry import TRANSFORMS
import numpy as np

@TRANSFORMS.register_module()
class AddScaleFactor:

    def __call__(self, results):

        # identity scale for both bbox and mask branches
        results['scale_factor'] = np.array([1., 1.], dtype=np.float32)

        # also set scale (MMDet expects this for bookkeeping)
        results['scale'] = (1., 1.)

        results['keep_ratio'] = True

        return results
