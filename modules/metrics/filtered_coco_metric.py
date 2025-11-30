from mmdet.evaluation import CocoMetric
from mmdet.registry import METRICS
import torch


@METRICS.register_module()
class FilteredCocoMetric(CocoMetric):
    def __init__(self, filter_classes, **kwargs):
        super().__init___()
        self.filter_classes = filter_classes
        
    def process(self, data_batch, data_samples):
        for data_sample in data_samples:
            