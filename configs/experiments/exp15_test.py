# ─────────────────────────────────────────────────────────────
# configs/experiments/exp15_convgamma_k5.py
# Test config — inherits from exp15 training config.
# Removes EarlyStoppingHook and PreprocessorMonitorHook
# which require train_loop and are not valid during test.
# ─────────────────────────────────────────────────────────────

_base_ = ['exp15_convgamma_k5.py']

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
        experiment_name='exp15_test',
        save_raw=True,
        save_preprocessed=True
    ),
]