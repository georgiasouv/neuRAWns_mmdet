from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class FreezeMultiDetectorHook(Hook):
    def __init__(self, 
                 priority='VERY_HIGH',
                 debug_mode=False,
                 check_updates=False):
        super().__init__()
        self.priority = priority
        self.debug_mode = debug_mode
        self.check_updates= check_updates
        self.initial_weights = {}
        self.initial_detector_weights = {}
        
    def before_train(self, runner): # Runs once AFTER model initialization & weight loading, but BEFORE the first training iteration.  
        model = runner.model
        
        if hasattr(model, 'module'): # Handle DDP wrapper if using distributed training
            model = model.module
            
        for param in model.parameters(): #1: Freeze EVERYTHING in the model
            param.requires_grad = False
        
        if hasattr(model, 'preprocessing'):
            for param in model.preprocessing.parameters():
                param.requires_grad = True
            
            total_params = sum(p.numel() for p in model.parameters())   # Count parameters
            trainable_params = sum(
                p.numel() for p in model.preprocessing.parameters() 
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
            
            if self.debug_mode:
                runner.logger.info("[DEBUG] Trainable modules:")
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        runner.logger.info( f"  ✓ {name}: {param.numel():,} params, shape={tuple(param.shape)}")

                
            if trainable_params > 50000:    # Sanity check 1 - fail fast if freezing didn't work
                raise RuntimeError(
                    f"Freezing failed! {trainable_params:,} trainable parameters "
                    f"(expected < 50,000). Check your model structure."
                )
            
            if trainable_params == 0:    # Sanity check 2 - make sure something is trainable
                raise RuntimeError(
                    "No trainable parameters found! "
                    "Check that model.preprocessing exists."
                )
        else:
            raise RuntimeError(
                "Cannot find preprocessing module on model! "
                "Check your backbone configuration. "
                "Expected: model.preprocessing to exist."
            )
        
        if self.check_updates:
            runner.logger.info("\n[DEBUG] Storing initial weights...")
            
            # Store preprocessor weights
            for name, param in model.preprocessing.named_parameters():
                if param.requires_grad:
                    self.initial_weights[name] = param.data.clone()
            
                # Store a few detector weights for comparison
                for name, param in list(model.detectors[0].named_parameters())[:2]:
                    self.initial_detector_weights[name] = param.data.clone()
                
                runner.logger.info("[DEBUG] Weights stored for comparison.\n")

    def after_train_epoch(self, runner):
        """Check if weights updated - only if enabled."""
        if not self.check_updates:
            return  # Skip entirely if not checking
        
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        
        runner.logger.info(
            f"\n{'='*70}\n"
            f"[DEBUG] Weight Update Check - Epoch {runner.epoch}\n"
            f"{'='*70}\n"
        )
        
        # Check preprocessor changed
        runner.logger.info("[DEBUG] Preprocessor (should CHANGE):")
        for name, param in model.preprocessing.named_parameters():
            if name in self.initial_weights:
                diff = (param.data - self.initial_weights[name]).abs().mean().item()
                status = '✓ CHANGED' if diff > 1e-6 else '✗ NOT CHANGED'
                runner.logger.info(f"  {name}: diff={diff:.8f} {status}")
        
        # Check detector frozen
        runner.logger.info("\n[DEBUG] Detector (should NOT CHANGE):")
        for name, param in list(model.detectors[0].named_parameters())[:2]:
            if name in self.initial_detector_weights:
                diff = (param.data - self.initial_detector_weights[name]).abs().mean().item()
                status = '✓ FROZEN' if diff < 1e-8 else '✗ CHANGED!'
                runner.logger.info(f"  {name}: diff={diff:.8f} {status}")
        
        runner.logger.info(f"{'='*70}\n")