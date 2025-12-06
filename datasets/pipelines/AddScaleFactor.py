from mmdet.registry import TRANSFORMS
import numpy as np

@TRANSFORMS.register_module()
class AddScaleFactor:

    def __call__(self, results):

        # Because the backbone halves resolution (packing)
        # the detector’s bbox outputs must be scaled ×2 back to original resolution.
        results['scale_factor'] = np.array([2., 2., 2., 2.], dtype=np.float32)

        results['scale'] = (1., 1.)
        results['keep_ratio'] = True

        return results
