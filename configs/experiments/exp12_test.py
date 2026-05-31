# ─────────────────────────────────────────────────────────────
# /experiments/exp12_pack4ch_gamma.py
# Test config — inherits from exp12 training config.
# Removes EarlyStoppingHook and PreprocessorMonitorHook
# which require train_loop and are not valid during test.
# ─────────────────────────────────────────────────────────────

_base_ = ['exp12_pack4ch_gamma.py']

custom_hooks = [
    dict(
        type='FreezeDetectorHook',
        debug_mode=False,
        check_updates=False,
        priority='VERY_HIGH'
    ),
    dict(
        type='SaveBatchImagesHook',
        save_dir='sample_images_test',
        experiment_name='exp12',
        save_raw=True,
        save_preprocessed=True
    ),
]