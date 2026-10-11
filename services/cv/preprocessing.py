"""
Computer Vision Preprocessing & Postprocessing Utilities
========================================================

Handles robust image ingestion, aspect-preserving letterboxing, normalization,
and bounding-box coordinate transformations across OpenCV, PIL, and raw file inputs.
"""

import base64
import io
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import yaml
except ImportError:
    yaml = None

from .schemas import BoundingBox

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "sensor_profiles.yaml"


def get_preprocessing_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Loads CV preprocessing configuration from config/sensor_profiles.yaml
    with safe defaults (enable_clahe=True, clip_limit=2.0, tile_grid=(8, 8)).
    """
    cfg_file = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    defaults = {
        "enable_clahe": True,
        "clip_limit": 2.0,
        "tile_grid": (8, 8)
    }
    if yaml and cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            prep = loaded.get("preprocessing", {})
            if isinstance(prep, dict):
                tg = prep.get("clahe_tile_grid", [8, 8])
                tile_tuple = (int(tg[0]), int(tg[1])) if isinstance(tg, (list, tuple)) and len(tg) >= 2 else (8, 8)
                return {
                    "enable_clahe": bool(prep.get("enable_clahe", True)),
                    "clip_limit": float(prep.get("clahe_clip_limit", 2.0)),
                    "tile_grid": tile_tuple
                }
        except Exception:
            pass
    return defaults


def _apply_clahe_numpy_pil_fallback(
    arr_rgb: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Pure NumPy/PIL fallback for CLAHE when OpenCV (cv2) is unavailable.
    Operates on the Y (luminance) channel in YCbCr color space using
    tile-grid contrast-limited histogram equalization with bilinear tile interpolation.
    """
    h, w = arr_rgb.shape[:2]
    ycbcr_img = Image.fromarray(arr_rgb, mode="RGB").convert("YCbCr")
    y_ch, cb_ch, cr_ch = ycbcr_img.split()
    L = np.asarray(y_ch, dtype=np.uint8)

    tx = max(1, min(w, int(tile_grid[0])))
    ty = max(1, min(h, int(tile_grid[1])))
    clip_val = max(0.1, float(clip_limit))

    y_edges = np.linspace(0, h, ty + 1, dtype=int)
    x_edges = np.linspace(0, w, tx + 1, dtype=int)
    luts = np.zeros((ty, tx, 256), dtype=np.float64)

    for r in range(ty):
        y0, y1 = y_edges[r], max(y_edges[r] + 1, y_edges[r + 1])
        for c in range(tx):
            x0, x1 = x_edges[c], max(x_edges[c] + 1, x_edges[c + 1])
            tile = L[y0:y1, x0:x1]
            n_pix = max(1, tile.size)
            hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)

            limit = max(1.0, clip_val * (n_pix / 256.0))
            excess = float(np.maximum(0.0, hist - limit).sum())
            hist = np.minimum(hist, limit) + (excess / 256.0)

            cdf = np.cumsum(hist)
            nonzero = cdf[cdf > 0]
            cdf_min = float(nonzero[0]) if nonzero.size > 0 else 0.0
            denom = max(1e-6, float(n_pix) - cdf_min)
            luts[r, c] = np.clip((cdf - cdf_min) / denom * 255.0, 0.0, 255.0)

    # Bilinear interpolation across tile centers
    cy = 0.5 * (y_edges[:-1] + y_edges[1:] - 1.0)
    cx = 0.5 * (x_edges[:-1] + x_edges[1:] - 1.0)

    ys = np.arange(h, dtype=np.float64)
    xs = np.arange(w, dtype=np.float64)

    if ty == 1:
        r0 = np.zeros(h, dtype=int)
        r1 = np.zeros(h, dtype=int)
        wy = np.zeros(h, dtype=np.float64)
    else:
        idx_y = np.clip(np.searchsorted(cy, ys, side="right") - 1, 0, ty - 2)
        r0 = idx_y
        r1 = idx_y + 1
        span_y = np.maximum(1e-6, cy[r1] - cy[r0])
        wy = np.clip((ys - cy[r0]) / span_y, 0.0, 1.0)

    if tx == 1:
        c0 = np.zeros(w, dtype=int)
        c1 = np.zeros(w, dtype=int)
        wx = np.zeros(w, dtype=np.float64)
    else:
        idx_x = np.clip(np.searchsorted(cx, xs, side="right") - 1, 0, tx - 2)
        c0 = idx_x
        c1 = idx_x + 1
        span_x = np.maximum(1e-6, cx[c1] - cx[c0])
        wx = np.clip((xs - cx[c0]) / span_x, 0.0, 1.0)

    v00 = luts[r0[:, None], c0[None, :], L]
    v01 = luts[r0[:, None], c1[None, :], L]
    v10 = luts[r1[:, None], c0[None, :], L]
    v11 = luts[r1[:, None], c1[None, :], L]

    top = (1.0 - wx[None, :]) * v00 + wx[None, :] * v01
    bot = (1.0 - wx[None, :]) * v10 + wx[None, :] * v11
    L_eq = np.clip(np.round((1.0 - wy[:, None]) * top + wy[:, None] * bot), 0, 255).astype(np.uint8)

    merged = Image.merge("YCbCr", (Image.fromarray(L_eq, mode="L"), cb_ch, cr_ch)).convert("RGB")
    return np.array(merged, dtype=np.uint8)


def apply_clahe(
    img_rgb: Union[np.ndarray, str, Path, bytes, Image.Image],
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
    force_fallback: bool = False
) -> np.ndarray:
    """
    Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE) to the luminance
    channel of an RGB image while preserving chromaticity.

    Uses OpenCV CIE LAB L-channel CLAHE when cv2 is available, falling back to a
    pure NumPy/PIL YCbCr bilinear tile-grid implementation when cv2 is absent.

    Args:
        img_rgb: Input RGB image (H, W, 3) uint8 array or loadable image input.
        clip_limit: Contrast clipping threshold (default 2.0).
        tile_grid: Grid size (cols, rows) for contextual tiles (default (8, 8)).
        force_fallback: If True, forces the NumPy/PIL fallback even if cv2 is installed.

    Returns:
        Contrast-enhanced RGB uint8 NumPy array of shape (H, W, 3).
    """
    if not isinstance(img_rgb, np.ndarray):
        arr, _ = load_image(img_rgb)
    else:
        arr = img_rgb
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        arr = np.ascontiguousarray(arr, dtype=np.uint8)

    if arr.size == 0:
        return arr

    tx = max(1, int(tile_grid[0]))
    ty = max(1, int(tile_grid[1]))
    clip_val = max(0.1, float(clip_limit))

    if HAS_CV2 and not force_fallback:
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_val, tileGridSize=(tx, ty))
        l_eq = clahe.apply(l_ch)
        lab_eq = cv2.merge((l_eq, a_ch, b_ch))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    return _apply_clahe_numpy_pil_fallback(arr, clip_limit=clip_val, tile_grid=(tx, ty))


def encode_image_base64(img_rgb: np.ndarray, fmt: str = "JPEG", quality: int = 90) -> str:
    """Encodes an RGB uint8 NumPy array into a base64 data URI string."""
    pil_img = Image.fromarray(np.ascontiguousarray(img_rgb, dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    save_kwargs = {"quality": quality} if fmt.upper() in ("JPEG", "JPG") else {}
    pil_img.save(buf, format=fmt.upper(), **save_kwargs)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() in ("JPEG", "JPG") else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


def load_image(
    image_input: Union[str, Path, bytes, np.ndarray, Image.Image]
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Ingests any supported image representation and converts it into a standard
    RGB NumPy array (height, width, channels) with dimensions (original_width, original_height).

    Returns:
        (image_rgb_array, (original_width, original_height))
    """
    # Case 1: File path
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image file does not exist: {p}")
        pil_img = Image.open(p).convert("RGB")
        arr = np.array(pil_img, dtype=np.uint8)
        w, h = pil_img.size
        return arr, (w, h)

    # Case 2: Raw bytes
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        arr = np.array(pil_img, dtype=np.uint8)
        w, h = pil_img.size
        return arr, (w, h)

    # Case 3: PIL Image
    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
        arr = np.array(pil_img, dtype=np.uint8)
        w, h = pil_img.size
        return arr, (w, h)

    # Case 4: NumPy array
    if isinstance(image_input, np.ndarray):
        arr = image_input
        # If grayscale, convert to 3-channel
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        # If RGBA, drop alpha
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        h, w = arr.shape[:2]
        return arr.astype(np.uint8), (w, h)

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def letterbox(
    image: np.ndarray,
    target_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resizes image to target_shape while preserving aspect ratio using padding.

    Returns:
        (letterboxed_image, scale_ratio, (pad_left, pad_top))
    """
    h, w = image.shape[:2]
    target_h, target_w = target_shape

    # Compute scale ratio
    ratio = min(target_w / w, target_h / h)
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))

    pad_w = (target_w - new_w) / 2.0
    pad_h = (target_h - new_h) / 2.0

    pad_left = int(round(pad_w - 0.1))
    pad_right = int(round(pad_w + 0.1))
    pad_top = int(round(pad_h - 0.1))
    pad_bottom = int(round(pad_h + 0.1))

    if HAS_CV2:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        letterboxed = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=color
        )
    else:
        pil_img = Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR)
        letterboxed_pil = Image.new("RGB", (target_w, target_h), color)
        letterboxed_pil.paste(pil_img, (pad_left, pad_top))
        letterboxed = np.array(letterboxed_pil, dtype=np.uint8)

    return letterboxed, ratio, (pad_left, pad_top)


def denormalize_box_from_letterbox(
    box: BoundingBox,
    ratio: float,
    padding: Tuple[float, float],
    original_dims: Tuple[int, int]
) -> BoundingBox:
    """
    Transforms bounding box coordinates from letterboxed target space back to original image space.
    """
    pad_x, pad_y = padding
    orig_w, orig_h = original_dims

    # Invert padding and scaling
    xmin = max(0.0, min(float(orig_w), (box.xmin - pad_x) / ratio))
    ymin = max(0.0, min(float(orig_h), (box.ymin - pad_y) / ratio))
    xmax = max(xmin + 1.0, min(float(orig_w), (box.xmax - pad_x) / ratio))
    ymax = max(ymin + 1.0, min(float(orig_h), (box.ymax - pad_y) / ratio))

    return BoundingBox(xmin=round(xmin, 1), ymin=round(ymin, 1), xmax=round(xmax, 1), ymax=round(ymax, 1))
