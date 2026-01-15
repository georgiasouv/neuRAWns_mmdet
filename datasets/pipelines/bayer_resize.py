from __future__ import annotations

import numpy as np
import cv2
import torch

from mmdet.registry import TRANSFORMS
from mmcv.transforms import BaseTransform


@TRANSFORMS.register_module()
class BayerResize(BaseTransform):
    """Bayer-aware resize for RGGB mosaics.

    Resizes by splitting the mosaic into four planes (R,G1,G2,B),
    resizing each plane, then re-interleaving. This avoids CFA mixing.
    """

    def __init__(
        self,
        scale=(1333, 800),          # (max_w, max_h) as in MMDet
        keep_ratio=True,
        interpolation_down="area",  # best for downscale
        interpolation_up="linear",  # fine for upscale
        force_even=True,            # must be even for RGGB re-interleave
    ):
        super().__init__()
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.force_even = force_even

        interp_map = {
            "area": cv2.INTER_AREA,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "nearest": cv2.INTER_NEAREST,
        }
        if interpolation_down not in interp_map:
            raise ValueError(f"Unknown interpolation_down={interpolation_down}")
        if interpolation_up not in interp_map:
            raise ValueError(f"Unknown interpolation_up={interpolation_up}")

        self.interp_down = interp_map[interpolation_down]
        self.interp_up = interp_map[interpolation_up]

    @staticmethod
    def _make_even(x: int) -> int:
        return x if (x % 2 == 0) else (x - 1)

    def _get_raw_and_key(self, results: dict):
        # MMDet usually uses 'img' before PackDetInputs; your loader might use 'inputs'
        if "img" in results:
            key = "img"
        elif "inputs" in results:
            key = "inputs"
        else:
            raise KeyError("BayerResize expects `img` or `inputs` in results.")

        raw = results[key]

        # Convert to numpy HxW float32 for OpenCV
        if isinstance(raw, torch.Tensor):
            t = raw
            # allow (1,H,W) or (H,W)
            if t.ndim == 3 and t.shape[0] == 1:
                t = t[0]
            if t.ndim != 2:
                raise ValueError(f"Expected mosaic tensor as (H,W) or (1,H,W), got {tuple(raw.shape)}")
            raw_np = t.detach().cpu().numpy().astype(np.float32, copy=False)
            return key, raw_np, True, raw.dtype, (raw.ndim == 3 and raw.shape[0] == 1)
        else:
            arr = np.asarray(raw)
            # allow (H,W,1) or (H,W)
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr2 = arr[..., 0]
                keep_hw1 = True
            else:
                arr2 = arr
                keep_hw1 = False
            if arr2.ndim != 2:
                raise ValueError(f"Expected mosaic array as (H,W) or (H,W,1), got {arr.shape}")
            raw_np = arr2.astype(np.float32, copy=False)
            return key, raw_np, False, None, keep_hw1

    def _compute_new_size(self, w: int, h: int):
        max_w, max_h = self.scale  # note order (w,h)

        if not self.keep_ratio:
            new_w, new_h = int(max_w), int(max_h)
        else:
            sf = min(max_w / w, max_h / h)
            new_w = int(round(w * sf))
            new_h = int(round(h * sf))

        if self.force_even:
            new_w = max(self._make_even(new_w), 2)
            new_h = max(self._make_even(new_h), 2)

        w_scale = new_w / w
        h_scale = new_h / h
        return new_w, new_h, w_scale, h_scale

    def transform(self, results: dict) -> dict:
        key, raw, was_torch, orig_dtype, keep_channel_dim = self._get_raw_and_key(results)

        h, w = raw.shape

        # Ensure valid RGGB grid
        if self.force_even and ((h % 2) != 0 or (w % 2) != 0):
            raw = raw[: self._make_even(h), : self._make_even(w)]
            h, w = raw.shape

        # Store original size if missing (MMDet uses these fields)
        results.setdefault("ori_shape", (h, w))

        new_w, new_h, w_scale, h_scale = self._compute_new_size(w, h)

        # No-op
        if new_w == w and new_h == h:
            results["img_shape"] = (h, w)
            results["scale_factor"] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            return results

        # Resize in plane domain
        ph, pw = new_h // 2, new_w // 2

        # Choose interpolation
        interp = self.interp_down if (new_w < w or new_h < h) else self.interp_up

        # Split RGGB planes
        R  = raw[0::2, 0::2]
        G1 = raw[0::2, 1::2]
        G2 = raw[1::2, 0::2]
        B  = raw[1::2, 1::2]

        # Resize each plane (OpenCV uses (width,height))
        Rr  = cv2.resize(R,  (pw, ph), interpolation=interp)
        G1r = cv2.resize(G1, (pw, ph), interpolation=interp)
        G2r = cv2.resize(G2, (pw, ph), interpolation=interp)
        Br  = cv2.resize(B,  (pw, ph), interpolation=interp)

        # Re-interleave into mosaic
        out = np.empty((new_h, new_w), dtype=np.float32)
        out[0::2, 0::2] = Rr
        out[0::2, 1::2] = G1r
        out[1::2, 0::2] = G2r
        out[1::2, 1::2] = Br

        # Write back preserving container type / rank conventions
        if was_torch:
            out_t = torch.from_numpy(out)
            if orig_dtype is not None and out_t.dtype != orig_dtype:
                out_t = out_t.to(orig_dtype)
            if keep_channel_dim:
                out_t = out_t.unsqueeze(0)  # (1,H,W)
            results[key] = out_t
        else:
            if keep_channel_dim:
                results[key] = out[..., None]  # (H,W,1)
            else:
                results[key] = out

        results["img_shape"] = (new_h, new_w)
        # MMDet expects 4 values for bbox scaling
        results["scale_factor"] = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)

        return results
