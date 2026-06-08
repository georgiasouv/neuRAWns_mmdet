# ─────────────────────────────────────────────────────────────
# modules/hooks/ensemble_guards.py
# Guard hooks for the multi-detector ensemble. These complement
# modules/hooks/ensemble_diagnostics.py (freeze / grad / stats):
#
#   NaNGuardHook              : kill the run at the FIRST non-finite
#                               loss or gradient, naming the culprit
#   PreprocessorUpdateHook    : is the preprocessor actually learning?
#                               (stall + explosion detection)
#   DetectorInputContractHook : after each detector's OWN
#                               DetDataPreprocessor runs, is the tensor
#                               it feeds its backbone in-distribution?
#
# Reads scratch attributes stashed by MultiDetectorWrapper /
# EnsembleTrainLoop:
#   model._last_per_detector_grads  (dict[name -> grad tensor])
#   model._last_det_input_stats     (dict[name -> dict of floats])
# If those are absent the hooks no-op silently (by design, so the
# default loop still runs).
# ─────────────────────────────────────────────────────────────
import math

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


def _unwrap(model):
    return model.module if hasattr(model, 'module') else model


@HOOKS.register_module()
class NaNGuardHook(Hook):
    """Halts training the moment any logged loss or any stashed
    per-detector gradient is non-finite (NaN or Inf).

    Why halt instead of warn: with three detectors sharing one
    preprocessor, a single NaN gradient from ONE detector corrupts the
    shared module's weights on the very next optimizer step, and every
    metric after that is garbage. Failing loudly with the detector's
    name turns a wasted overnight run into a 1-minute diagnosis.
    """

    priority = 'VERY_HIGH'

    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None):
        bad = []

        # 1) losses: `outputs` is the log_vars dict the loop returns.
        #    Keys are namespaced 'detname.loss_xxx', so a hit names
        #    the detector for free.
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    if not torch.isfinite(v).all():
                        bad.append(f'loss[{k}]')
                elif isinstance(v, (int, float)):
                    if not math.isfinite(v):
                        bad.append(f'loss[{k}]')

        # 2) per-detector gradients on the shared enhanced tensor.
        model = _unwrap(runner.model)
        grads = getattr(model, '_last_per_detector_grads', None) or {}
        for name, g in grads.items():
            if g is not None and not torch.isfinite(g).all():
                bad.append(f'grad[{name}]')

        if bad:
            raise RuntimeError(
                f'[NaNGuard] non-finite values at iter {runner.iter}: '
                f'{", ".join(bad)}. Usual suspects, in order: (1) LR too '
                f'high for the preprocessor, (2) enhanced output left '
                f'[0,255] and a detector loss blew up (check '
                f'enhanced/max in the log just before this iter), '
                f'(3) a degenerate batch (empty GT). The checkpoint on '
                f'disk predates this iter and is safe to resume from.')


@HOOKS.register_module()
class PreprocessorUpdateHook(Hook):
    """Tracks whether the preprocessor's weights are actually moving.

    Logs two scalars every `log_interval` iters:
      preproc/step_norm  : ||theta_now - theta_prev_interval||
                           (how big was the recent movement)
      preproc/rel_drift  : ||theta_now - theta_init|| / ||theta_init||
                           (cumulative movement since train start)

    Warns when:
      - rel_drift is still ~0 after `stall_warn_iters` -> the module is
        NOT learning (LR=0 for its param group, grads detached, or the
        optimizer wasn't given its params)
      - rel_drift exceeds `explode_rel` -> instability (LR too high)
    """

    def __init__(self, log_interval=100, stall_warn_iters=500,
                 stall_tol=1e-7, explode_rel=2.0):
        self.log_interval = log_interval
        self.stall_warn_iters = stall_warn_iters
        self.stall_tol = stall_tol
        self.explode_rel = explode_rel
        self._init_vec = None
        self._prev_vec = None
        self._stall_warned = False

    @torch.no_grad()
    def _flat_params(self, model):
        ps = [p.detach().flatten() for p in model.preprocessor.parameters()]
        return torch.cat(ps).clone() if ps else None

    def before_train(self, runner):
        model = _unwrap(runner.model)
        if not hasattr(model, 'preprocessor'):
            return
        self._init_vec = self._flat_params(model)
        self._prev_vec = self._init_vec.clone() if self._init_vec is not None else None

    @torch.no_grad()
    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None):
        if self._init_vec is None:
            return
        if not self.every_n_train_iters(runner, self.log_interval):
            return

        model = _unwrap(runner.model)
        cur = self._flat_params(model)

        step_norm = (cur - self._prev_vec).norm().item()
        rel_drift = ((cur - self._init_vec).norm() /
                     (self._init_vec.norm() + 1e-12)).item()
        self._prev_vec = cur

        runner.message_hub.update_scalar('preproc/step_norm', step_norm)
        runner.message_hub.update_scalar('preproc/rel_drift', rel_drift)

        if (not self._stall_warned
                and runner.iter >= self.stall_warn_iters
                and rel_drift < self.stall_tol):
            self._stall_warned = True
            runner.logger.warning(
                f'[PreprocessorUpdate] rel_drift={rel_drift:.2e} after '
                f'{runner.iter} iters — the preprocessor has effectively '
                f'not moved. Check: (1) its params are in the optimizer '
                f'(paramwise_cfg / optim_wrapper), (2) its LR is nonzero, '
                f'(3) grads reach it (grad/norm_* hooks show nonzero).')

        if rel_drift > self.explode_rel:
            runner.logger.warning(
                f'[PreprocessorUpdate] rel_drift={rel_drift:.2f} exceeds '
                f'{self.explode_rel} — weights have moved more than '
                f'{self.explode_rel * 100:.0f}% of their init norm. If '
                f'enhanced stats are also drifting, lower the LR.')


@HOOKS.register_module()
class DetectorInputContractHook(Hook):
    """Checks the tensor each detector's backbone ACTUALLY receives,
    i.e. AFTER that detector's own DetDataPreprocessor (COCO mean/std)
    has run on your [0,255] enhanced image (Option A).

    Reads model._last_det_input_stats, which the wrapper stashes inside
    _detector_loss (see the 4-line wrapper patch).

    Expectation under Option A: roughly zero-mean, unit-ish std.
    Dark scenes legitimately sit negative (a near-black image lands at
    about (0-110)/57 = -1.9), so the warn bands are deliberately loose:
      mean outside [-2.5, 2.5]  -> suspicious
      std  outside [0.2, 4.0]   -> suspicious
    A persistent violation means the enhanced image left the [0,255]
    contract or the wrong normalization ran.
    """

    def __init__(self, log_interval=50, mean_band=(-2.5, 2.5),
                 std_band=(0.2, 4.0)):
        self.log_interval = log_interval
        self.mean_band = mean_band
        self.std_band = std_band

    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None):
        if not self.every_n_train_iters(runner, self.log_interval):
            return
        model = _unwrap(runner.model)
        stats = getattr(model, '_last_det_input_stats', None)
        if not stats:
            return

        for name, s in stats.items():
            runner.message_hub.update_scalar(f'det_in/{name}_mean', s['mean'])
            runner.message_hub.update_scalar(f'det_in/{name}_std', s['std'])

            ok_mean = self.mean_band[0] <= s['mean'] <= self.mean_band[1]
            ok_std = self.std_band[0] <= s['std'] <= self.std_band[1]
            if not (ok_mean and ok_std):
                runner.logger.warning(
                    f'[DetectorInputContract] {name} backbone input is '
                    f'out of band: mean={s["mean"]:.2f} std={s["std"]:.2f} '
                    f'min={s["min"]:.2f} max={s["max"]:.2f}. Under Option '
                    f'A this means the enhanced image is no longer '
                    f'[0,255]-scaled before {name}\'s DetDataPreprocessor '
                    f'— check enhanced/max and the *255 output scaling.')