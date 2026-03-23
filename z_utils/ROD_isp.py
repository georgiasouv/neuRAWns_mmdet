import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

BIT8 = 2**8
BIT16 = 2**16
BIT24 = 2**24

def read_raw_24b(file_path, img_shape=(1, 1, 1856, 2880), read_type=np.uint8):
    raw_data = np.fromfile(file_path, dtype=read_type)
    raw_data = raw_data.astype(np.float32)
    raw_data = raw_data[0::3] + raw_data[1::3] * BIT8 + raw_data[2::3] * BIT16
    raw_data = raw_data.reshape(img_shape).astype(np.float32).squeeze()
    return raw_data

def normalise_with_percentile(raw, low=0.5, high=99.9):
    low_val = np.percentile(raw, low)
    high_val = np.percentile(raw, high)
    return np.clip((raw - low_val) / (high_val - low_val), 0, 1)

def demosaic_rggb_safe(raw_float):
    raw_16bit = np.clip(raw_float * 65535, 0, 65535).astype(np.uint16)
    rgb = cv2.demosaicing(raw_16bit, cv2.COLOR_BayerRG2BGR)
    rgb = rgb.astype(np.float32) / 65535.0
    return rgb

def auto_white_balance(img):
    avg_rgb = np.mean(img.reshape(-1, 3), axis=0)
    gains = avg_rgb.mean() / (avg_rgb + 1e-6)
    return img * gains[None, None, :]

def warm_tint(img, r_boost=1.04, b_drop=0.96):
    gains = np.array([b_drop, 1.0, r_boost])
    return img * gains[None, None, :]

def tone_map_log(img, exposure=2.0):
    img = exposure * img
    return np.log1p(img) / np.log1p(exposure)

def highlight_rolloff(img, knee=0.9):
    return np.where(img > knee, knee + (1 - knee) * np.tanh((img - knee) / (1 - knee)), img)

def edge_enhance(img, strength=0.4):
    blur = cv2.bilateralFilter((img * 255).astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=75)
    detail = img - blur.astype(np.float32) / 255.0
    return np.clip(img + strength * detail, 0, 1)

def gamma_and_output(img, gamma=2.2):
    img = np.power(np.clip(img, 0, 1), 1.0 / gamma)
    return (img * 255).astype(np.uint8)

def aesthetic_pipeline(
    path,
    img_shape,
    apply_warm_tint=True,
    apply_highlight_rolloff=True,
    apply_edge_enhance=True,
    save_as="jpg",
    jpeg_quality=95
):
    # === Load RAW and Normalise ===
    raw = read_raw_24b(path, img_shape)
    raw = normalise_with_percentile(raw, low=0.5, high=99.9)

    # === Demosaic and White Balance ===
    rgb = demosaic_rggb_safe(raw)
    rgb = auto_white_balance(rgb)
    rgb = np.clip(rgb, 0, 1)

    # === Optional: Colour tint to correct blue cast ===
    if apply_warm_tint:
        rgb = warm_tint(rgb)

    # === Tone Mapping ===
    rgb = tone_map_log(rgb, exposure=2.0)

    # === Optional: Highlight rolloff to suppress bloom ===
    if apply_highlight_rolloff:
        rgb = highlight_rolloff(rgb)

    # === Optional: Sharpening ===
    if apply_edge_enhance:
        rgb = edge_enhance(rgb, strength=0.4)

    # === Final Output ===
    output_img = gamma_and_output(rgb)

    # === Save ===
    filename = os.path.splitext(os.path.basename(path))[0]
    if save_as == "jpg":
        out_path = f"{filename}_isp.jpg"
        cv2.imwrite(out_path, output_img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    elif save_as == "png":
        out_path = f"{filename}_isp.png"
        cv2.imwrite(out_path, output_img)
    else:
        raise ValueError("Unsupported format. Use 'jpg' or 'png'.")

    print(f"Saved output to: {out_path}")
    return output_img


# img_shape = (1, 1, 1856, 2880)

# output = aesthetic_pipeline(
#     path=img_path,
#     img_shape=img_shape,
#     apply_warm_tint=True,
#     apply_highlight_rolloff=True,
#     apply_edge_enhance=True,
#     save_as="jpg",
#     jpeg_quality=95
# )

# plt.imshow(output)
# plt.axis("off")
# plt.title("Publication-Ready ISP Output: night-06979")
# plt.show()

