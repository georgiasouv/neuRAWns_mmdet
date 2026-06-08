# ─────────────────────────────────────────────────────────────
# modules/hooks/preproc_observability.py
# Observability for the spatially-adaptive tone-mapping module
# (LocalAffine / LocalCurve / LocalHybrid) and the fixed
# normalisation front-end (NormaliseLinear / NormaliseLog).
#
#   ToneParamDiagnosticHook : is the 8x8 grid actually spatially
#                             adaptive, or has it collapsed to a
#                             global operator?
#   NormInputStatsHook      : stats of the normalised RAW input;
#                             warns if AMP is on while night signal
#                             sits below fp16's normal range
#   IdentityInitProbeHook   : one-shot check at iter 1 that the
#                             zero-init heads really start near
#                             identity (output ~ 255 * input scale)
#
# Contracts (stashed by wrapper / module, all detached):
#   model._last_norm_input               (wrapper._enhance)
#   model._last_enhanced                 (EnsembleTrainLoop, existing)
#   model.preprocessor._last_tone_params (module forward: dict[str->Tensor])
# Hooks no-op silently if a stash is absent.
# ─────────────────────────────────────────────────────────────
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

FP16_MIN_NORMAL = 6.1e-5  # smallest normal float16; below this = subnormal


def _unwrap(model):
    return model.module if hasattr(model, 'module') else model


@HOOKS.register_module()
class ToneParamDiagnosticHook(Hook):
    """Logs statistics of the predicted tone parameters.

    For every tensor in model.preprocessor._last_tone_params:
      tone/{k}_mean, tone/{k}_std, tone/{k}_min, tone/{k}_max
    and, when the tensor has trailing spatial dims (the 8x8 grid):
      tone/{k}_spatial_std  — std ACROSS grid cells, averaged over
                              batch and channels.

    How to read spatial_std:
      ~0 at init        : expected (identity / zero-init heads)
      grows in training : the module is exploiting spatial structure —
                          the headline justification for the grid design
      stays ~0 forever  : the grid collapsed to a global op; you are
                          paying the thumbnail-CNN cost for nothing.
                          That is a finding (compare against LocalHybrid
                          which is global-by-design), not just a bug —
                          but verify LR/grad flow before believing it.
    """

    def __init__(self, log_interval=50):
        self.log_interval = log_interval

    @torch.no_grad()
    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None):
        if not self.every_n_train_iters(runner, self.log_interval):
            return
        model = _unwrap(runner.model)
        pre = getattr(model, 'preprocessor', None)
        params = getattr(pre, '_last_tone_params', None)
        if not params:
            return

        for k, t in params.items():
            if t is None:
                continue
            t = t.float()
            runner.message_hub.update_scalar(f'tone/{k}_mean', t.mean().item())
            runner.message_hub.update_scalar(f'tone/{k}_std', t.std().item())
            runner.message_hub.update_scalar(f'tone/{k}_min', t.min().item())
            runner.message_hub.update_scalar(f'tone/{k}_max', t.max().item())
            # spatial variance across the grid: flatten last two dims
            if t.dim() >= 3 and t.shape[-1] > 1 and t.shape[-2] > 1:
                flat = t.flatten(-2)                       # (..., H*W)
                sp_std = flat.std(dim=-1).mean().item()    # std over cells
                runner.message_hub.update_scalar(f'tone/{k}_spatial_std',
                                                 sp_std)


@HOOKS.register_module()
class NormInputStatsHook(Hook):
    """Stats of the normalised RAW tensor the tone module receives.

    Logs: norm_in/mean, norm_in/max, norm_in/frac_subnormal_fp16
    (fraction of nonzero values below fp16's smallest normal number).

    Under NormaliseLinear, night signal sits near 5e-5 — BELOW the
    fp16 normal range. If AMP is enabled, that signal degrades to
    subnormals before the module sees it. This hook warns once if
    AMP appears active while a meaningful fraction of the input is
    in that danger zone. Fix: keep the normalisation + tone module
    in fp32 (autocast exclusion), or use NormaliseLog for AMP runs.
    """

    def __init__(self, log_interval=50, subnormal_warn_frac=0.05):
        self.log_interval = log_interval
        self.subnormal_warn_frac = subnormal_warn_frac
        self._amp_warned = False

    @torch.no_grad()
    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None):
        if not self.every_n_train_iters(runner, self.log_interval):
            return
        model = _unwrap(runner.model)
        x = getattr(model, '_last_norm_input', None)
        if x is None:
            return
        x = x.float()

        nz = x[x > 0]
        frac_sub = ((nz < FP16_MIN_NORMAL).float().mean().item()
                    if nz.numel() else 0.0)

        runner.message_hub.update_scalar('norm_in/mean', x.mean().item())
        runner.message_hub.update_scalar('norm_in/max', x.max().item())
        runner.message_hub.update_scalar('norm_in/frac_subnormal_fp16',
                                         frac_sub)

        amp_on = 'Amp' in type(runner.optim_wrapper).__name__
        if (amp_on and frac_sub > self.subnormal_warn_frac
                and not self._amp_warned):
            self._amp_warned = True
            runner.logger.warning(
                f'[NormInputStats] AMP is active and {frac_sub:.0%} of '
                f'nonzero input values are below fp16 normal range '
                f'({FP16_MIN_NORMAL:.1e}). Under NormaliseLinear this is '
                f'your night signal being quantised away BEFORE the tone '
                f'module. Keep the front-end in fp32 or switch this run '
                f'to NormaliseLog.')


@HOOKS.register_module()
class IdentityInitProbeHook(Hook):
    """One-shot probe on the first training iteration.

    All three module variants are designed to start at identity
    (zero-init heads), with output on a 0-255 scale. So at iter 1:
        enhanced.mean()  ≈  255 * norm_input.mean()
    (approximately — the 4->3 channel mix reweights channels, so we
    check the ratio is O(1), not exact equality).

    If the ratio is far from 1, the init is NOT identity: a head
    wasn't zero-init'd, out_scale didn't apply, or an activation is
    saturating at init. Catching this at iter 1 saves you from
    misreading the first epochs as 'slow learning'.
    """

    def __init__(self, ratio_band=(0.2, 5.0)):
        self.ratio_band = ratio_band
        self._done = False

    @torch.no_grad()
    def after_train_iter(self, runner, batch_idx, data_batch=None,
                         outputs=None):
        if self._done:
            return
        self._done = True
        model = _unwrap(runner.model)
        x = getattr(model, '_last_norm_input', None)
        y = getattr(model, '_last_enhanced', None)
        if x is None or y is None:
            runner.logger.warning(
                '[IdentityInitProbe] stashes missing — is '
                'EnsembleTrainLoop running and is _last_norm_input '
                'stashed in _enhance()?')
            return

        ratio = y.float().mean().item() / (255.0 * x.float().mean().item()
                                           + 1e-12)
        runner.logger.info(
            f'[IdentityInitProbe] iter 1: enhanced.mean / '
            f'(255*input.mean) = {ratio:.3f} '
            f'(in [0,255]: enhanced mean={y.float().mean().item():.2f}, '
            f'max={y.float().max().item():.2f})')
        lo, hi = self.ratio_band
        if not (lo <= ratio <= hi):
            runner.logger.warning(
                f'[IdentityInitProbe] ratio {ratio:.3f} outside '
                f'[{lo}, {hi}] — the module is NOT starting near '
                f'identity. Check zero-init of the prediction heads, '
                f'the out_scale=255 application, and that the 1x1 mix '
                f'bias is zero-init.')