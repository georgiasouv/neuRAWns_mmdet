from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class FreezeDetectorHook(Hook):
    def __init__(self, priority='VERY_HIGH'):
        super().__init__()
        self.priority = priority
        
    def before_train(self, runner): # Runs once AFTER model initialization & weight loading, but BEFORE the first training iteration.  
        model = runner.model
        
        if hasattr(model, 'module'): # Handle DDP wrapper if using distributed training
            model = model.module
            
        for param in model.parameters(): #1: Freeze EVERYTHING in the model
            param.requires_grad = False
        
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'preprocessor'):  # 2: Unfreeze ONLY the preprocessing module
            for param in model.backbone.preprocessor.parameters():
                param.requires_grad = True
            
            total_params = sum(p.numel() for p in model.parameters())   # Count parameters
            trainable_params = sum(
                p.numel() for p in model.backbone.preprocessor.parameters() 
                if p.requires_grad
            )
            frozen_params = total_params - trainable_params
            percentage = 100 * trainable_params / total_params
            
            runner.logger.info(
                f"\n{'='*70}\n"
                f"DETECTOR FREEZING COMPLETE\n"
                f"{'='*70}\n"
                f"Total parameters:     {total_params:>15,}\n"
                f"Trainable parameters: {trainable_params:>15,}\n"
                f"Frozen parameters:    {frozen_params:>15,}\n"
                f"Percentage trainable: {percentage:>15.6f}%\n"
                f"{'='*70}\n"
            )
            
            # List trainable modules for verification
            runner.logger.info("Trainable modules:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    runner.logger.info(f"  - {name}: {param.numel():,} params")
            runner.logger.info(f"{'='*70}\n")
            
            if trainable_params > 50000:    # Sanity check 1 - fail fast if freezing didn't work
                raise RuntimeError(
                    f"Freezing failed! {trainable_params:,} trainable parameters "
                    f"(expected < 50,000). Check your model structure."
                )
            
            
            if trainable_params == 0:    # Sanity check 2 - make sure something is trainable
                raise RuntimeError(
                    "No trainable parameters found! "
                    "Check that model.backbone.preprocessor exists."
                )
        else:
            raise RuntimeError(
                "Cannot find preprocessor in model.backbone! "
                "Check your backbone configuration. "
                "Expected: model.backbone.preprocessor to exist."
            )

    