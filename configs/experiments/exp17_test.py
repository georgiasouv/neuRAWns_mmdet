# ─────────────────────────────────────────────────────────────
# configs/experiments/exp17_convpower.py
# Test config — inherits from exp17 training config.
# Removes EarlyStoppingHook and PreprocessorMonitorHook
# which require train_loop and are not valid during test.
# ─────────────────────────────────────────────────────────────

_base_ = ['exp17_convpower.py']

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
        experiment_name='exp17_test',
        save_raw=True,
        save_preprocessed=True
    ),
]