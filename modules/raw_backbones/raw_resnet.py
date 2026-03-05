import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.backbones import ResNet

@MODELS.register_module()
class RAWResNet(nn.Module):
    def __init__(self,
                 debug_mode=False,
                 preprocess_cfg=dict(
                     type='RAWPreprocess_v1',
                     in_channels=1,
                     out_channels=3,
                     norm_threshold=0.99),
                 **resnet_kwargs):
        super().__init__()
        self.preprocessor = MODELS.build(preprocess_cfg)
        self.resnet = ResNet(**resnet_kwargs)
        self.debug_mode = debug_mode
        self._first_forward = True
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Remap checkpoint keys from backbone.X to backbone.resnet.X"""
        
        # Create remapped state dict
        remapped = {}
        keys_to_remove = []
        
        for key, value in state_dict.items():
            if key.startswith(prefix):
                # Remove the prefix to get the local key
                local_key = key[len(prefix):]
                
                # If it's a resnet key without 'resnet.' prefix, add it
                if not local_key.startswith('resnet.') and not local_key.startswith('preprocessor.'):
                    new_key = prefix + 'resnet.' + local_key
                    remapped[new_key] = value
                    keys_to_remove.append(key)
        
        # Update state_dict with remapped keys
        for key in keys_to_remove:
            del state_dict[key]
        state_dict.update(remapped)
        
        # Call parent implementation
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    
    def forward(self, x):
        if self.debug_mode and self._first_forward:
            print(f"\n{'='*70}")
            print("[DEBUG] RAW_ResNet Forward - First Batch")
            print(f"{'='*70}")
            print(f"Input shape:  {x.shape}")  
            print(f"Input range:  [{x.min():.4f}, {x.max():.4f}]")
        
        x = self.preprocessor(x)
        
        if self.debug_mode and self._first_forward:
            print(f"\nAfter preprocessing:")
            print(f"Shape: {x.shape}")
            print(f"Range: [{x.min():.4f}, {x.max():.4f}]")
            print(f"Mean per channel: R={x[:,0].mean():.4f}, G={x[:,1].mean():.4f}, B={x[:,2].mean():.4f}")
            print(f"Std per channel:  R={x[:,0].std():.4f}, G={x[:,1].std():.4f}, B={x[:,2].std():.4f}")
            print(f"Overall mean: {x.mean():.4f}, Overall std: {x.std():.4f}")
            print(f"\n{'='*70}")
            self._first_forward = False
            
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.resnet(x)
        return out

    def get_preprocessed_for_visualisation(self, x):
        x = self.preprocessor(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = x * std + mean
        return x
            
    def init_weights(self):
        self.resnet.init_weights()

