"""
Computer Vision Preprocessing & Postprocessing Utilities
========================================================

Handles robust image ingestion, aspect-preserving letterboxing, normalization,
and bounding-box coordinate transformations across OpenCV, PIL, and raw file inputs.
"""

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

from .schemas import BoundingBox


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
