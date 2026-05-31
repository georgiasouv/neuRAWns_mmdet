# ─────────────────────────────────────────────────────────────
# configs/experiments/exp13_convgammagain.py
# Test config — inherits from exp13 training config.
# Removes EarlyStoppingHook and PreprocessorMonitorHook
# which require train_loop and are not valid during test.
# ─────────────────────────────────────────────────────────────

_base_ = ['exp13_convgammagain.py']

custom_hooks = [
    dict(
        type='FreezeDetectorHook',
        debug_mode=False,
        check_updates=False,
        priority='VERY_HIGH'
    ),
    dict(
        type='SaveBatchImagesHook',
        save_dir='sample_image',
        experiment_name='exp13_test',
        save_raw=True,
        save_preprocessed=True
    ),
]