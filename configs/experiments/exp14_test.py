# ─────────────────────────────────────────────────────────────
# configs/experiments/exp14_convgamma_k1.py
# Test config — inherits from exp14 training config.
# Removes EarlyStoppingHook and PreprocessorMonitorHook
# which require train_loop and are not valid during test.
# ─────────────────────────────────────────────────────────────

_base_ = ['exp14_convgamma_k1.py']

custom_hooks = [
    dict(
        type='FreezeDetectorHook',
        debug_mode=False,
        check_updates=False,
        priority='VERY_HIGH'
    ),
    dict(
        type='SaveBatchImagesHook',
        save_dir='sample_images',
        experiment_name='exp14_test',
        save_raw=True,
        save_preprocessed=True
    ),
]