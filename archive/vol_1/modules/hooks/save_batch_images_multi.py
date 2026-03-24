import os
import numpy as np
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import torch
from PIL import Image

@HOOKS.register_module()
class SaveBatchImagesHook_Multi(Hook):
    def __init__(self,
                 save_dir='sample_images',
                 experiment_name='exp002',
                 save_raw=False,
                 save_preprocessed=True):
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.save_raw = save_raw
        self.save_preprocessed = save_preprocessed
        self.first_batch_saved = False
        
    def before_train_epoch(self, runner):
        """Reset flag at the start of each epoch."""
        self.first_batch_saved = False
        
    def after_train_iter(self,
                         runner,
                         batch_idx,
                         data_batch=None,
                         outputs=None):
        
        if self.first_batch_saved:
            return
        
        if batch_idx == 0:
            self._save_batch(runner, data_batch)
            self.first_batch_saved = True
            
    def _save_batch(self,
                    runner,  # the runner contains training state
                    data_batch): # a dict contianing the batch data
        
        current_epoch = runner.epoch + 1
        
        epoch_dir = os.path.join(self.save_dir, self.experiment_name, f'epoch_{current_epoch}')
        os.makedirs(epoch_dir, exist_ok=True)
        
        inputs = data_batch['inputs'] # list of images in the batch
        
        # Save each image in the batch
        batch_size = len(inputs)
        for i in range(batch_size):
            img = inputs[i]
            
            # Save raw input 
            if self.save_raw:
                raw_array = img.cpu().numpy() # convert pytorch tensor to numpy array
                raw_path = os.path.join(epoch_dir, f'raw_img_{i}.npy')
                np.save(raw_path, raw_array)
           
            
            if self.save_preprocessed:
                with torch.no_grad():
                    model = runner.model
                    if hasattr(model, 'module'):
                        model = model.module
                    
                    device = next(model.parameters()).device
                    
                    # Ensure even dimensions for Bayer packing
                    _, h, w = img.shape
                    if h % 2 != 0 or w % 2 != 0:
                        # Crop to even dimensions
                        h_even = h - (h % 2)
                        w_even = w - (w % 2)
                        img = img[:, :h_even, :w_even]
                    img_input = img.unsqueeze(0).to(device)
                    
                    if hasattr(model, 'preprocessing'):
                        preprocessed = model.preprocessing(img_input)
                        preprocessed = preprocessed[0]
                    
                     # Convert to numpy for saving
                        preprocessed_np = preprocessed.cpu().numpy()  # [3, H, W]
                        preprocessed_np = np.transpose(preprocessed_np, (1, 2, 0))  # [H, W, 3]
                        
                        # Clip to valid range and convert to uint8
                        preprocessed_np = np.clip(preprocessed_np, 0, 1) * 255
                        preprocessed_np = preprocessed_np.astype(np.uint8)
                        
                        # Save as PNG
                        img_pil = Image.fromarray(preprocessed_np, mode='RGB')
                        png_path = os.path.join(epoch_dir, f'preprocessed_img_{i}.png')
                        img_pil.save(png_path)
            
        runner.logger.info(
            f'Saved {batch_size} images from epoch {current_epoch} to {epoch_dir}')
                
        
        
        
        
        