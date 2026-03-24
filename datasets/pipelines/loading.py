import numpy as np
from mmdet.registry import TRANSFORMS
import mmcv
import rawpy
from pathlib import Path

RGB_EXT = ['.jpg', '.jpeg', '.png', '.bmp']
RAW_EXT = ['.arw', '.nef', '.dng']
ROD_EXT = ['.raw']

@TRANSFORMS.register_module()
class LoadRAWImageFromFile:
    def __init__(self):
        pass
    
    def __call__(self, results):
        img_path_str = results['img_path']
        img_path = Path(img_path_str)
        suffix = img_path.suffix.lower()

        if suffix in RGB_EXT:
            img = mmcv.imread(img_path_str)
        elif suffix in RAW_EXT:
            with rawpy.imread(img_path_str) as raw:
                img = raw.raw_image.copy()
        elif suffix in ROD_EXT:
            BIT8, BIT16 = 2**8, 2**16
            img = np.fromfile(img_path_str, dtype=np.uint8)
            img = img.astype(np.float32)
            img = img[0::3] + img[1::3] * BIT8 + img[2::3] * BIT16
            img = img.reshape((1856, 2880))
        else:
            raise ValueError(f'Unsupported image format: {suffix}')

        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
            
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results
    
@TRANSFORMS.register_module()
class PackBayer:
    """Pack single-channel Bayer [H,W,1] → 3-channel [H/2,W/2,3]
    by extracting RGGB channels and averaging the two greens.
    Must be applied BEFORE any spatial transforms (Resize, Flip, Crop).
    """
    def __call__(self, results):
        img = results['img']          # [H, W, 1] float32
        img = img.squeeze(2)          # [H, W]
        H, W = img.shape

        R  = img[0::2, 0::2]
        G1 = img[0::2, 1::2]
        G2 = img[1::2, 0::2]
        B  = img[1::2, 1::2]
        G  = 0.5 * (G1 + G2)

        packed = np.stack([R, G, B], axis=2)  # [H/2, W/2, 3]

        results['img'] = packed
        results['img_shape'] = packed.shape
        results['ori_shape'] = packed.shape
        return results