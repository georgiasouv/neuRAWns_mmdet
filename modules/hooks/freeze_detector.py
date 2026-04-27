from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class FreezeDetectorHook(Hook):
    def __init__(self, priority='VERY_HIGH', debug_mode=False, check_updates=False):
        super().__init__()
        self.priority = priority
        self.debug_mode = debug_mode
        self.check_updates = check_updates
        self.initial_weights = {}
        self.initial_detector_weights = {}

    def before_train(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        # Validate structure early
        if not (hasattr(model, 'data_preprocessor') and
                hasattr(model.data_preprocessor, 'raw_preprocessor')):
            raise RuntimeError(
                "Expected model.data_preprocessor.raw_preprocessor to exist. "
                "Check your RAWDetDataPreprocessor config."
            )

        # Freeze everything, then unfreeze only the learnable preprocessor
        for param in model.parameters():
            param.requires_grad = False
        for param in model.data_preprocessor.raw_preprocessor.parameters():
            param.requires_grad = True

        # Parameter counts
        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params    = total_params - trainable_params

        runner.logger.info(
            f"\n{'='*60}\n"
            f"DETECTOR FREEZING COMPLETE\n"
            f"{'='*60}\n"
            f"Total:      {total_params:>12,}\n"
            f"Trainable:  {trainable_params:>12,}  ({100*trainable_params/total_params:.4f}%)\n"
            f"Frozen:     {frozen_params:>12,}\n"
            f"{'='*60}\n"
        )

        # Sanity checks
        if trainable_params == 0:
            raise RuntimeError("No trainable parameters — preprocessor has no parameters.")
        if trainable_params > 50_000:
            raise RuntimeError(
                f"Freezing failed: {trainable_params:,} trainable params (expected <50,000)."
            )

        if self.debug_mode:
            runner.logger.info("[DEBUG] Trainable parameters:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    runner.logger.info(f"  ✓ {name}: {param.numel():,} {tuple(param.shape)}")

        if self.check_updates:
            for name, param in model.data_preprocessor.raw_preprocessor.named_parameters():
                self.initial_weights[name] = param.data.clone()
            for name, param in list(model.backbone.named_parameters())[:2]:
                self.initial_detector_weights[name] = param.data.clone()
            runner.logger.info("[DEBUG] Initial weights stored.")

    def after_train_epoch(self, runner):
        if not self.check_updates:
            return

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        runner.logger.info(f"\n{'='*60}\nWeight Update Check — Epoch {runner.epoch}\n{'='*60}")

        runner.logger.info("Preprocessor (should CHANGE):")
        for name, param in model.data_preprocessor.raw_preprocessor.named_parameters():
            if name in self.initial_weights:
                diff = (param.data - self.initial_weights[name]).abs().mean().item()
                runner.logger.info(f"  {'✓ CHANGED' if diff > 1e-6 else '✗ STATIC':12} {name}  diff={diff:.2e}")

        runner.logger.info("Detector backbone (should NOT CHANGE):")
        for name, param in list(model.backbone.named_parameters())[:2]:
            if name in self.initial_detector_weights:
                diff = (param.data - self.initial_detector_weights[name]).abs().mean().item()
                runner.logger.info(f"  {'✓ FROZEN' if diff < 1e-8 else '✗ CHANGED!':12} {name}  diff={diff:.2e}")

        runner.logger.info(f"{'='*60}\n")