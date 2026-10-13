"""
preprocessing.py - Image Preprocessing Pipeline for Canvas Input.

This module converts arbitrary drawings (from mouse, touch, or uploaded images)
into the standardized 28x28 grayscale format expected by the MNIST ANN model.

Pipeline Stages:
    1. Input normalization: Convert RGBA/RGB/Grayscale to 2D numpy array.
    2. Empty canvas check: Gracefully reject blank or near-blank drawings.
    3. Background polarity handling: Invert light-on-dark vs dark-on-light
       so that foreground stroke is bright (~255) and background is dark (0).
    4. Bounding box cropping: Locate digit stroke boundaries.
    5. Aspect-ratio preserving resize: Scale digit to fit in ~20x20 bounding area.
    6. Center-of-mass / Bounding centering: Place digit within a 28x28 frame.
    7. Pixel normalization: Scale intensities to [0.0, 1.0].
    8. Reshape: Output both (28, 28) for preview and (1, 784) for ANN inference.
"""

from typing import Tuple, Optional
import numpy as np
from PIL import Image


class EmptyCanvasError(ValueError):
    """Raised when the input canvas contains no discernible drawn digit."""
    pass


def is_canvas_empty(
    image_array: np.ndarray,
    min_stroke_pixels: int = 15,
    stroke_threshold: float = 25.0,
) -> bool:
    """
    Determines whether a canvas drawing contains sufficient drawn content.

    Args:
        image_array: 2D or 3D numpy array representing canvas pixels.
        min_stroke_pixels: Minimum number of stroke pixels required.
        stroke_threshold: Intensity difference threshold from the background.

    Returns:
        True if the canvas is empty or contains negligible content, else False.
    """
    if image_array is None or image_array.size == 0:
        return True

    # If 3D RGBA/RGB, extract alpha or luminosity
    if image_array.ndim == 3:
        if image_array.shape[2] == 4:
            # Check if all pixels are fully transparent
            alpha = image_array[:, :, 3]
            if np.all(alpha == 0):
                return True
            # For RGBA where strokes have alpha > 0
            visible_mask = alpha > 20
            if np.sum(visible_mask) < min_stroke_pixels:
                return True

        # Convert to grayscale luminosity
        rgb = image_array[:, :, :3].astype(np.float32)
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    else:
        gray = image_array.astype(np.float32)

    # Check variation / range
    intensity_range = np.ptp(gray)  # max - min
    if intensity_range < stroke_threshold:
        return True

    # Contrast against background (check if foreground pixels exist)
    # Estimate background from corners
    corners = np.array([
        gray[0, 0], gray[0, -1],
        gray[-1, 0], gray[-1, -1]
    ])
    bg_val = np.median(corners)
    diff = np.abs(gray - bg_val)

    return int(np.sum(diff > stroke_threshold)) < min_stroke_pixels


def extract_stroke_intensity(image_array: np.ndarray) -> np.ndarray:
    """
    Converts arbitrary image input to a 2D float array where the background is 0.0
    and the digit stroke is positive (> 0.0, up to 255.0).

    Args:
        image_array: RGBA, RGB, or Grayscale image array.

    Returns:
        2D numpy array of shape (H, W) with stroke intensities [0.0, 255.0].
    """
    arr = np.asarray(image_array, dtype=np.float32)

    if arr.ndim == 3:
        if arr.shape[2] == 4:
            # RGBA format
            r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
            gray = 0.299 * r + 0.587 * g + 0.114 * b

            # Check if canvas uses alpha as stroke (common in drawing canvases)
            alpha_range = np.ptp(a)
            if alpha_range > 50 and np.mean(a < 50) > 0.5:
                # Strokes are drawn on transparent background
                stroke = np.where(a > 20, 255.0 - gray if np.mean(gray) > 128 else gray, 0.0)
                return np.clip(stroke, 0.0, 255.0)

            # Normal RGB with alpha channel fully opaque
            # Detect background polarity from border
            border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
            bg_mean = np.mean(border)

            if bg_mean > 127.0:
                # Light background (e.g., white canvas with dark pen) -> invert
                stroke = 255.0 - gray
            else:
                # Dark background (e.g., black canvas with white pen) -> keep
                stroke = gray

            return np.clip(stroke, 0.0, 255.0)

        elif arr.shape[2] == 3:
            # RGB format
            gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
            if np.mean(border) > 127.0:
                stroke = 255.0 - gray
            else:
                stroke = gray
            return np.clip(stroke, 0.0, 255.0)

    # 2D Grayscale
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if np.mean(border) > 127.0:
        stroke = 255.0 - arr
    else:
        stroke = arr

    return np.clip(stroke, 0.0, 255.0)


def compute_bounding_box(
    stroke_image: np.ndarray,
    threshold: float = 20.0,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Finds the bounding box (min_row, min_col, max_row, max_col) of non-background strokes.

    Args:
        stroke_image: 2D stroke intensity image where stroke > 0.
        threshold: Minimum intensity to count as a stroke pixel.

    Returns:
        Tuple of (min_row, min_col, max_row, max_col) or None if below threshold.
    """
    rows = np.any(stroke_image > threshold, axis=1)
    cols = np.any(stroke_image > threshold, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]

    return int(min_row), int(min_col), int(max_row), int(max_col)


def center_of_mass(image: np.ndarray) -> Tuple[float, float]:
    """
    Computes the center of mass (cy, cx) of pixel intensities in a 2D image.

    Args:
        image: 2D float numpy array.

    Returns:
        Tuple (center_y, center_x).
    """
    total_mass = np.sum(image)
    if total_mass == 0:
        return image.shape[0] / 2.0, image.shape[1] / 2.0

    y_indices, x_indices = np.indices(image.shape)
    cy = np.sum(y_indices * image) / total_mass
    cx = np.sum(x_indices * image) / total_mass

    return float(cy), float(cx)


def shift_image(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """
    Shifts a 2D image by (dy, dx) using integer translation with zero-padding.

    Args:
        image: 2D numpy array.
        dy: Vertical shift in pixels.
        dx: Horizontal shift in pixels.

    Returns:
        Shifted 2D image of identical shape.
    """
    shift_y = int(round(dy))
    shift_x = int(round(dx))

    h, w = image.shape
    shifted = np.zeros_like(image)

    src_y_start = max(0, -shift_y)
    src_y_end = min(h, h - shift_y)
    src_x_start = max(0, -shift_x)
    src_x_end = min(w, w - shift_x)

    dst_y_start = max(0, shift_y)
    dst_y_end = min(h, h + shift_y)
    dst_x_start = max(0, shift_x)
    dst_x_end = min(w, w + shift_x)

    if (src_y_end > src_y_start) and (src_x_end > src_x_start):
        shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image[
            src_y_start:src_y_end, src_x_start:src_x_end
        ]

    return shifted


def preprocess_canvas_image(
    canvas_image,
    target_box_size: int = 20,
    final_size: int = 28,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline matching the MNIST standard:
        1. Validates canvas content (raises EmptyCanvasError if blank).
        2. Normalizes foreground/background convention (white digit on black).
        3. Crops to stroke bounding box.
        4. Scales to fit within 20x20 box preserving aspect ratio.
        5. Centers in 28x28 frame using center-of-mass shift.
        6. Normalizes pixel values to [0.0, 1.0].
        7. Returns both (28, 28) preview array and (1, 784) flattened vector.

    Args:
        canvas_image: numpy array or PIL Image from drawing canvas.
        target_box_size: Bounding box size for the digit (MNIST standard is 20).
        final_size: Final output dimension (MNIST standard is 28).

    Returns:
        Tuple:
            - processed_28x28: np.ndarray shape (28, 28) with values in [0.0, 1.0].
            - flattened_784: np.ndarray shape (1, 784) ready for model.predict().

    Raises:
        EmptyCanvasError: If canvas is blank or contains no discernible digit.
    """
    if canvas_image is None:
        raise EmptyCanvasError("Canvas is empty. Please draw a digit before predicting.")

    # Convert PIL Image to numpy array if applicable
    if isinstance(canvas_image, Image.Image):
        img_arr = np.array(canvas_image)
    else:
        img_arr = np.array(canvas_image)

    # 1. Validate empty canvas
    if is_canvas_empty(img_arr):
        raise EmptyCanvasError("No digit detected on canvas. Please draw a digit (0-9).")

    # 2. Extract stroke intensity (background = 0, stroke = [0, 255])
    stroke_img = extract_stroke_intensity(img_arr)

    # 3. Suppress weak background noise
    stroke_img = np.where(stroke_img < 20.0, 0.0, stroke_img)

    # 4. Compute bounding box
    bbox = compute_bounding_box(stroke_img, threshold=20.0)
    if bbox is None:
        raise EmptyCanvasError("No digit strokes detected above noise threshold.")

    min_row, min_col, max_row, max_col = bbox
    crop = stroke_img[min_row : max_row + 1, min_col : max_col + 1]

    crop_h, crop_w = crop.shape
    if crop_h <= 1 or crop_w <= 1:
        raise EmptyCanvasError("Drawn stroke is too small to recognize. Please draw a full digit.")

    # 5. Aspect-ratio preserving resize into (target_box_size x target_box_size)
    pil_crop = Image.fromarray(crop.astype(np.uint8), mode="L")

    scale = float(target_box_size) / max(crop_h, crop_w)
    new_w = max(1, min(target_box_size, int(round(crop_w * scale))))
    new_h = max(1, min(target_box_size, int(round(crop_h * scale))))

    # Lanczos filter provides smooth anti-aliased stroke downsampling
    resized_pil = pil_crop.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    resized_digit = np.array(resized_pil, dtype=np.float32)

    # 6. Place resized digit in 28x28 frame
    canvas_28 = np.zeros((final_size, final_size), dtype=np.float32)
    start_y = (final_size - new_h) // 2
    start_x = (final_size - new_w) // 2
    canvas_28[start_y : start_y + new_h, start_x : start_x + new_w] = resized_digit

    # 7. Fine-tune centering by Center of Mass (LeCun MNIST standard)
    cy, cx = center_of_mass(canvas_28)
    target_center = (final_size - 1) / 2.0  # 13.5
    dy = target_center - cy
    dx = target_center - cx

    # Only shift if shift is reasonable (within 3 pixels) to prevent digit from clipping edges
    if abs(dy) < 4.0 and abs(dx) < 4.0:
        canvas_28 = shift_image(canvas_28, dy, dx)

    # 8. Normalize to [0.0, 1.0]
    processed_28x28 = np.clip(canvas_28 / 255.0, 0.0, 1.0).astype(np.float32)

    # 9. Flatten into shape (1, 784)
    flattened_784 = processed_28x28.reshape(1, 784)

    return processed_28x28, flattened_784
