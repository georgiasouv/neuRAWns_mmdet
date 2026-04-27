import torch
from mmdet.registry import MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor


@MODELS.register_module()
class RAWDetDataPreprocessor(DetDataPreprocessor):

    def __init__(self, preprocessor_cfg: dict, **kwargs):
        super().__init__(**kwargs)
        self.raw_preprocessor = MODELS.build(preprocessor_cfg)

    def forward(self, data: dict, training: bool = False) -> dict:
        data = self.cast_data(data)
        inputs = data['inputs']
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs, dim=0)
            inputs = self.raw_preprocessor(inputs)
            data['inputs'] = list(inputs)
        return super().forward(data, training=training)