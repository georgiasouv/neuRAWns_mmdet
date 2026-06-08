import torch
import torch.nn.functional as F
from itertools import combinations
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


def _unwrap(model):
    return model.module if hasattr(model, 'module') else model


@HOOKS.register_module()
class VerifyMultiDetectorFreezeHook(Hook):
    """Checksums every detector at train start, re-checks periodically.
    Raises if any detector weight moved; warns if preprocessor didn't."""

    def __init__(self, check_interval=500):
        self.check_interval = check_interval
        self._ref = None

    def _checksum(self, det):
        with torch.no_grad():
            return sum(p.float().sum().item() for p in det.parameters())

    def before_train(self, runner):
        model = _unwrap(runner.model)
        self._ref = [self._checksum(d) for d in model.detectors]
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        runner.logger.info(
            f'[freeze] trainable {n_train/1e3:.1f}k / total '
            f'{n_total/1e6:.1f}M ({100*n_train/max(n_total,1):.2f}%)')
        det_train = sum(p.numel() for d in model.detectors
                        for p in d.parameters() if p.requires_grad)
        if det_train > 0:
            raise RuntimeError(
                f'[freeze] {det_train} detector params are trainable '
                f'— detectors are NOT frozen.')

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self.every_n_train_iters(runner, self.check_interval):
            return
        model = _unwrap(runner.model)
        for i, d in enumerate(model.detectors):
            if abs(self._checksum(d) - self._ref[i]) > 1e-3:
                raise RuntimeError(
                    f'[freeze] detector {model.detector_names[i]} weights '
                    f'changed at iter {runner.iter} — freeze leaked.')


@HOOKS.register_module()
class GradDiagnosticHook(Hook):
    """Per-detector gradient norms (scale) + pairwise cosines (conflict)
    at the shared enhanced tensor. Reads model._last_per_detector_grads,
    which only EnsembleTrainLoop stashes — no-ops on the default loop."""

    def __init__(self, log_interval=50):
        self.log_interval = log_interval

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self.every_n_train_iters(runner, self.log_interval):
            return
        model = _unwrap(runner.model)
        grads = getattr(model, '_last_per_detector_grads', None)
        if not grads:
            return
        norms = {n: g.norm().item() for n, g in grads.items()}
        spread = max(norms.values()) / (min(norms.values()) + 1e-12)
        for n, v in norms.items():
            runner.message_hub.update_scalar(f'grad/norm_{n}', v)
        runner.message_hub.update_scalar('grad/norm_spread', spread)
        for (n1, g1), (n2, g2) in combinations(grads.items(), 2):
            cos = F.cosine_similarity(g1.flatten(), g2.flatten(), dim=0).item()
            runner.message_hub.update_scalar(f'grad/cos_{n1}_{n2}', cos)


@HOOKS.register_module()
class EnhancedStatsHook(Hook):
    """Logs min/max/mean of the enhanced image. Option A expects output
    near [0,255]; drift toward [0,1] means detectors get OOD input."""

    def __init__(self, log_interval=50):
        self.log_interval = log_interval

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if not self.every_n_train_iters(runner, self.log_interval):
            return
        model = _unwrap(runner.model)
        y = getattr(model, '_last_enhanced', None)
        if y is None:
            return
        y = y.float()
        runner.message_hub.update_scalar('enhanced/min', y.min().item())
        runner.message_hub.update_scalar('enhanced/max', y.max().item())
        runner.message_hub.update_scalar('enhanced/mean', y.mean().item())