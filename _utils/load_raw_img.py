import rawpy
import numpy as np
from pathlib import Path
from PIL import Image

RGB_EXT = ['.jpg', '.jpeg', '.png', '.bmp']
RAW_EXT = ['.arw', '.nef', '.dng']
ROD_EXT = ['.raw'] 


def load_raw(filepath):
    img_path = Path(filepath)
    suffix = img_path.suffix.lower()
    if suffix in RGB_EXT:
        img = Image.open(img_path)
        img = np.array(img)
    elif suffix in RAW_EXT:
        with rawpy.imread(filepath) as raw:
            img = raw.raw_image.copy()    
    elif suffix in ROD_EXT:
            BIT8, BIT16 = 2**8, 2**16
            img = np.fromfile(filepath, dtype=np.uint8)
            img = img.astype(np.float32)
            img = img[0::3] + img[1::3] * BIT8 + img[2::3] * BIT16
            img = img.reshape((1856, 2880))
    else:
        raise ValueError(f'Unsupported image format: {suffix}')  

    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    return img
        