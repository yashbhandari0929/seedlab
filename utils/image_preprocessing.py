"""
SeedLab AI — MNIST-Compliant Image Preprocessing Pipeline.
Converts user-drawn canvas strokes into standardized 28x28 MNIST feature vectors.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
from scipy.ndimage import center_of_mass, shift


def preprocess_canvas_image(
    canvas_image_data: np.ndarray,
    target_dim: Tuple[int, int] = (28, 28),
    digit_box_size: int = 20,
    intensity_threshold: int = 25
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool, Dict[str, Any]]:
    """
    Transforms raw canvas RGBA drawing to match true MNIST distribution:
    1. Grayscale extraction & blank canvas validation.
    2. Bounding box cropping.
    3. Aspect-ratio preserving resize to fit inside digit_box_size (20x20).
    4. Centering within target_dim (28x28) canvas.
    5. Center of Mass alignment using scipy.ndimage.
    6. Normalization to [0.0, 1.0] and flattening to (1, 784).

    Args:
        canvas_image_data: RGBA numpy array of shape (H, W, 4) from st_canvas.
        target_dim: Final image dimensions, default (28, 28).
        digit_box_size: Bounding box target dimension, default 20.
        intensity_threshold: Minimum pixel intensity considered part of stroke (0-255).

    Returns:
        (processed_28x28, flattened_784, is_blank, metadata_dict)
    """
    if canvas_image_data is None:
        return None, None, True, {"error": "No image data provided."}

    # Extract Grayscale from RGBA
    # In st_canvas with black background and white strokes:
    # strokes have RGB ~ 255 and Alpha ~ 255.
    if canvas_image_data.ndim == 3 and canvas_image_data.shape[2] == 4:
        rgb = canvas_image_data[:, :, :3]
        alpha = canvas_image_data[:, :, 3].astype(np.float32) / 255.0
        # Grayscale from RGB weighted by alpha
        gray = np.dot(rgb, [0.2989, 0.5870, 0.1140]) * alpha
        gray = gray.astype(np.uint8)
    elif canvas_image_data.ndim == 3 and canvas_image_data.shape[2] == 3:
        gray = np.dot(canvas_image_data, [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    else:
        gray = canvas_image_data.astype(np.uint8)

    # Check for blank / empty canvas
    # Sum of pixels above threshold or max pixel
    max_val = np.max(gray)
    active_pixels = np.sum(gray > intensity_threshold)
    if max_val < intensity_threshold or active_pixels < 25:
        return None, None, True, {
            "reason": "Canvas is blank or stroke is too faint.",
            "max_intensity": int(max_val),
            "active_pixels": int(active_pixels)
        }

    # Locate Bounding Box
    rows = np.any(gray > intensity_threshold, axis=1)
    cols = np.any(gray > intensity_threshold, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Ensure valid bounding box
    if rmax <= rmin or cmax <= cmin:
        return None, None, True, {"reason": "Could not extract valid bounding box."}

    # Crop digit
    cropped = gray[rmin:rmax + 1, cmin:cmax + 1]
    crop_h, crop_w = cropped.shape

    # Aspect-ratio preserving resize to digit_box_size (20px)
    pil_crop = Image.fromarray(cropped)
    if crop_h > crop_w:
        new_h = digit_box_size
        new_w = max(1, int(round((crop_w / crop_h) * digit_box_size)))
    else:
        new_w = digit_box_size
        new_h = max(1, int(round((crop_h / crop_w) * digit_box_size)))

    resized_digit = pil_crop.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    resized_arr = np.array(resized_digit, dtype=np.float32)

    # Place resized digit into target 28x28 black canvas
    canvas_28 = np.zeros(target_dim, dtype=np.float32)
    start_y = (target_dim[0] - new_h) // 2
    start_x = (target_dim[1] - new_w) // 2
    canvas_28[start_y:start_y + new_h, start_x:start_x + new_w] = resized_arr

    # Center of Mass Alignment (Classic MNIST normalization)
    cy, cx = center_of_mass(canvas_28)
    if not (np.isnan(cy) or np.isnan(cx)):
        target_center_y, target_center_x = (target_dim[0] - 1) / 2.0, (target_dim[1] - 1) / 2.0
        shift_y = target_center_y - cy
        shift_x = target_center_x - cx
        # Shift with spline interpolation order 1 (linear) to avoid ring artifacts
        canvas_28 = shift(canvas_28, [shift_y, shift_x], order=1, mode='constant', cval=0.0)

    # Clean boundaries & clip
    canvas_28 = np.clip(canvas_28, 0.0, 255.0)

    # Normalize to [0.0, 1.0]
    normalized_28 = canvas_28 / 255.0
    flattened_784 = normalized_28.reshape(1, 784).astype(np.float32)

    metadata = {
        "is_blank": False,
        "crop_box": (int(cmin), int(rmin), int(cmax), int(rmax)),
        "resized_shape": (new_w, new_h),
        "center_of_mass": (float(cy) if not np.isnan(cy) else None, float(cx) if not np.isnan(cx) else None),
        "min_val": float(normalized_28.min()),
        "max_val": float(normalized_28.max())
    }

    return normalized_28, flattened_784, False, metadata
