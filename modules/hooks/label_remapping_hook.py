from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import numpy as np

@HOOKS.register_module()
class LabelRemappingHook(Hook):
    """Remap 80-class COCO predictions to 5-class subset before evaluation."""
    
    def __init__(self, coco_to_local_map):
        """
        Args:
            coco_to_local_map: dict mapping COCO class indices to local indices
                               e.g., {0: 0, 1: 1, 2: 2, 6: 3, 7: 4}
        """
        super().__init__()
        self.coco_to_local = coco_to_local_map
    
    def after_val_iter(self, runner, batch_idx, data_batch, outputs):
        """Remap labels in validation outputs."""
        for output in outputs:
            if 'labels' in output:
                labels = output['labels']
                # Only keep predictions for our 5 classes
                mask = np.isin(labels, list(self.coco_to_local.keys()))
                output['labels'] = np.array([self.coco_to_local[l] for l in labels[mask]])
                output['bboxes'] = output['bboxes'][mask]
                output['scores'] = output['scores'][mask]