import os
import numpy as np
from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import torch
from PIL import Image

@HOOKS.register_module()
class SaveBatchImagesHook(Hook):
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
            
    def _save_batch(self, runner, data_batch):
        current_epoch = runner.epoch + 1
        epoch_dir = os.path.join(self.save_dir, self.experiment_name, f'epoch_{current_epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        device = next(model.parameters()).device

        inputs = data_batch['inputs']

        for i, img in enumerate(inputs):
            if self.save_raw:
                np.save(os.path.join(epoch_dir, f'raw_img_{i}.npy'), img.cpu().numpy())

            if self.save_preprocessed:
                with torch.no_grad():
                    img_input = img.unsqueeze(0).float().to(device)

                    # Run through learnable preprocessor only (not mean/std norm)
                    preprocessed = model.data_preprocessor.raw_preprocessor(img_input)
                    preprocessed = preprocessed[0]  # [C, H, W]

                    # Normalise to [0, 255] for visualisation
                    p = preprocessed.cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
                    p = p - p.min()
                    if p.max() > 0:
                        p = p / p.max()
                    p = (p * 255).astype(np.uint8)

                    img_pil = Image.fromarray(p, mode='RGB')
                    img_pil.save(os.path.join(epoch_dir, f'preprocessed_img_{i}.png'))

        runner.logger.info(f'Saved {len(inputs)} images from epoch {current_epoch} to {epoch_dir}')
            
            
            
        