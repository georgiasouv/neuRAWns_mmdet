from mmengine.hooks import Hook
from mmdet.registry import HOOKS


@HOOKS.register_module()
class PreprocessorMonitorHook(Hook):
    """
    Logs gradient norms, weight norms, output statistics,
    and learnable scalar values for the preprocessor module only.
    """
    def __init__(self, log_every_n_steps=50):
        self.log_every_n_steps = log_every_n_steps

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        if batch_idx % self.log_every_n_steps != 0:
            return

        preprocessor = runner.model.data_preprocessor.raw_preprocessor
        log_dict = {}

        # ── Weight and gradient norms ──────────────────────────────────
        for name, param in preprocessor.named_parameters():
            log_dict[f'weight_norm/{name}'] = param.norm().item()
            if param.grad is not None:
                log_dict[f'grad_norm/{name}'] = param.grad.norm().item()

        # ── Learnable scalars (only relevant from exp16/17 onward) ─────
        if hasattr(preprocessor, 'alpha'):
            log_dict['learned/alpha'] = preprocessor.alpha.item()
        if hasattr(preprocessor, 'gamma'):
            log_dict['learned/gamma'] = preprocessor.gamma.item()

        runner.visualizer.add_scalars(log_dict)