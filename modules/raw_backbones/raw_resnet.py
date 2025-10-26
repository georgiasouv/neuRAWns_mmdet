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
        self._first_forward = True  # Flag to print only once, not spam terminal
    
    def forward(self, x):
        if self.debug_mode and self._first_forward:
            print(f"\n{'='*70}")
            print("[DEBUG] RAW_ResNet Forward - First Batch")
            print(f"{'='*70}")
            print(f"Input shape:  {x.shape}")  
            print(f"Input range:  [{x.min():.4f}, {x.max():.4f}]")
        
        x = self.preprocessor(x) #   x.shape (B, 1, H, W) OR (B, H, W) --> (B, 3, H/2, W/2)
        
        if self.debug_mode and self._first_forward:
            print(f"\nAfter preprocessing:")
            print(f"Shape: {x.shape}")
            print(f"Range: [{x.min():.4f}, {x.max():.4f}]")
            print(f"Mean per channel: R={x[:,0].mean():.4f}, G={x[:,1].mean():.4f}, B={x[:,2].mean():.4f}")
            print(f"Std per channel:  R={x[:,0].std():.4f}, G={x[:,1].std():.4f}, B={x[:,2].std():.4f}")
            print(f"Overall mean: {x.mean():.4f}, Overall std: {x.std():.4f}")
            print(f"\n{'='*70}")
            self._first_forward = False
            
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False) # Upsample back to full resolution --> (B, 3, H, W)
        out = self.resnet(x)
        return out

    def get_preprocessed_for_visualisation(self, x):
        x = self.preprocessor(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False ) 
        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) # Undo ImageNet normalization for natural-looking visualization
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = x * std + mean  # Denormalize
        return x
            
    def init_weights(self):
        self.resnet.init_weights()