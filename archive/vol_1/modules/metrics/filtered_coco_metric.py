from mmdet.evaluation import CocoMetric
from mmdet.registry import METRICS
import numpy as np
import torch

@METRICS.register_module()
class FilteredCocoMetric(CocoMetric):
    """CocoMetric that filters and remaps predictions from 80-class to 5-class."""
    
    def __init__(self, coco_to_local_map, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coco_to_local = coco_to_local_map
        print(f"FilteredCocoMetric initialized with mapping: {coco_to_local_map}")
    
    def process(self, data_batch, data_samples):
        """Process predictions before they reach the evaluator."""
        for data_sample in data_samples:
            pred = data_sample['pred_instances']
            labels = pred['labels'].cpu().numpy()
            
            # Keep only predictions for our 5 COCO classes
            mask = np.isin(labels, list(self.coco_to_local.keys()))
            
            # Remap to local indices
            remapped_labels = np.array([self.coco_to_local[l] for l in labels[mask]])
            
            # Convert to torch tensors on correct device
            mask_tensor = torch.from_numpy(mask).to(pred['labels'].device)
            remapped_tensor = torch.from_numpy(remapped_labels).to(pred['labels'].device)
            
            # Update predictions
            pred['labels'] = remapped_tensor
            pred['bboxes'] = pred['bboxes'][mask_tensor]
            pred['scores'] = pred['scores'][mask_tensor]
        
        super().process(data_batch, data_samples)