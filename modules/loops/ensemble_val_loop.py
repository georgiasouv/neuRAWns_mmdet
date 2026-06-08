# ─────────────────────────────────────────────────────────────
# modules/loops/ensemble_val_loop.py
# Validation loop for MultiDetectorWrapper.
#
# The default ValLoop evaluates ONE detector (whatever eval_detector
# points at). But the thesis claim is a VECTOR of mAPs — the shared
# preprocessor must serve every paradigm at once — so validation must
# produce per-detector metrics every val epoch. This loop runs the
# val dataloader once PER detector, prefixing each metric set with
# the detector name, plus a mean-across-detectors headline scalar.
#
# Cost: validation time scales with the number of detectors (3x).
# If that hurts, raise val_interval or point val_dataloader at a
# fixed subset — do NOT drop back to single-detector validation,
# or dominance by one detector becomes invisible until test time.
#
# Config:
#   val_cfg = dict(type='EnsembleValLoop')
# and add 'modules.loops.ensemble_val_loop' to custom_imports.
# NOTE for checkpointing: with prefixed keys, save_best must name a
# prefixed metric (e.g. 'ensemble/mean_mAP' from this loop).
# ─────────────────────────────────────────────────────────────
from mmengine.registry import LOOPS
from mmengine.runner.loops import ValLoop


def _unwrap(model):
    return model.module if hasattr(model, 'module') else model


@LOOPS.register_module()
class EnsembleValLoop(ValLoop):

    def run(self):
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        model = _unwrap(self.runner.model)
        if not hasattr(model, 'detector_names'):
            # Not the ensemble wrapper (e.g. a baseline run reusing this
            # config) — fall back to stock behaviour.
            return super().run()

        all_metrics = {}
        original = getattr(model, 'eval_detector', model.detector_names[0])

        for name in model.detector_names:
            model.eval_detector = name
            self.runner.logger.info(f'[EnsembleValLoop] validating '
                                    f'through detector: {name}')
            for idx, data_batch in enumerate(self.dataloader):
                self.run_iter(idx, data_batch)
            metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
            for k, v in metrics.items():
                all_metrics[f'{name}/{k}'] = v

        model.eval_detector = original

        # Headline scalar: mean mAP across detectors. This is the
        # number that summarises "one artifact serving all paradigms"
        # — and the right target for save_best.
        map_keys = [k for k in all_metrics
                    if k.endswith('mAP') and 'mAP_' not in k]
        if map_keys:
            mean_map = sum(float(all_metrics[k]) for k in map_keys) / len(map_keys)
            all_metrics['ensemble/mean_mAP'] = mean_map
            spread = (max(float(all_metrics[k]) for k in map_keys)
                      - min(float(all_metrics[k]) for k in map_keys))
            all_metrics['ensemble/mAP_spread'] = spread

        self.runner.call_hook('after_val_epoch', metrics=all_metrics)
        self.runner.call_hook('after_val')
        return all_metrics