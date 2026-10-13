"""
test_preprocessing.py - Unit Tests for Canvas Image Preprocessing.

Tests validation, bounding box extraction, center-of-mass centering,
stroke inversion, and output dimensional consistency.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
from PIL import Image

from preprocessing import (
    is_canvas_empty,
    extract_stroke_intensity,
    compute_bounding_box,
    center_of_mass,
    shift_image,
    preprocess_canvas_image,
    EmptyCanvasError,
)


def create_synthetic_digit_canvas(
    bg_color: int = 0,
    fg_color: int = 255,
    size: int = 280,
    stroke_thickness: int = 15,
) -> np.ndarray:
    """Creates a synthetic drawing canvas containing a vertical stroke ('1')."""
    canvas = np.full((size, size, 4), bg_color, dtype=np.uint8)
    canvas[:, :, 3] = 255  # fully opaque

    # Draw vertical bar in center
    mid = size // 2
    canvas[50:230, mid - stroke_thickness : mid + stroke_thickness, :3] = fg_color
    return canvas


def test_is_canvas_empty_on_blank():
    """Verifies that empty, uniform, or transparent canvases are detected."""
    # None input
    assert is_canvas_empty(None) is True

    # Empty array
    assert is_canvas_empty(np.zeros((0, 0))) is True

    # Uniform black canvas
    black = np.zeros((280, 280, 4), dtype=np.uint8)
    black[:, :, 3] = 255
    assert is_canvas_empty(black) is True

    # Uniform white canvas
    white = np.full((280, 280, 4), 255, dtype=np.uint8)
    assert is_canvas_empty(white) is True

    # Completely transparent canvas
    transparent = np.zeros((280, 280, 4), dtype=np.uint8)
    assert is_canvas_empty(transparent) is True


def test_is_canvas_empty_with_drawn_content():
    """Verifies that canvas with drawn stroke is detected as not empty."""
    canvas = create_synthetic_digit_canvas(bg_color=0, fg_color=255)
    assert is_canvas_empty(canvas) is False


def test_stroke_inversion_for_light_background():
    """Verifies that black strokes on white canvas are properly inverted."""
    # White canvas (255) with black stroke (0)
    white_bg_canvas = np.full((100, 100, 3), 255, dtype=np.uint8)
    white_bg_canvas[40:60, 40:60] = 0

    stroke = extract_stroke_intensity(white_bg_canvas)
    # Background should be near 0
    assert np.mean(stroke[:10, :10]) == 0.0
    # Stroke should be near 255
    assert np.mean(stroke[45:55, 45:55]) == 255.0


def test_stroke_preservation_for_dark_background():
    """Verifies that white strokes on black canvas are preserved."""
    # Black canvas (0) with white stroke (255)
    black_bg_canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    black_bg_canvas[40:60, 40:60] = 255

    stroke = extract_stroke_intensity(black_bg_canvas)
    assert np.mean(stroke[:10, :10]) == 0.0
    assert np.mean(stroke[45:55, 45:55]) == 255.0


def test_compute_bounding_box():
    """Verifies accurate bounding box coordinates of stroke."""
    img = np.zeros((100, 100), dtype=np.float32)
    # Draw a 20x30 rectangle from row 30 to 50, col 40 to 70
    img[30:51, 40:71] = 255.0

    bbox = compute_bounding_box(img, threshold=20.0)
    assert bbox is not None
    min_row, min_col, max_row, max_col = bbox
    assert min_row == 30
    assert max_row == 50
    assert min_col == 40
    assert max_col == 70


def test_center_of_mass():
    """Verifies center of mass calculation for symmetric and shifted mass."""
    img = np.zeros((50, 50), dtype=np.float32)
    img[20:30, 20:30] = 1.0  # Center is at (24.5, 24.5)

    cy, cx = center_of_mass(img)
    assert pytest.approx(cy, abs=0.1) == 24.5
    assert pytest.approx(cx, abs=0.1) == 24.5


def test_shift_image():
    """Verifies 2D translation shifts pixels accurately."""
    img = np.zeros((10, 10), dtype=np.float32)
    img[4, 4] = 1.0

    shifted = shift_image(img, dy=2, dx=3)
    assert shifted[6, 7] == 1.0
    assert shifted[4, 4] == 0.0


def test_preprocess_canvas_image_shapes_and_ranges():
    """Verifies end-to-end preprocessing produces (28, 28) and (1, 784) arrays in [0, 1]."""
    canvas = create_synthetic_digit_canvas(bg_color=0, fg_color=255)

    processed_28, flat_784 = preprocess_canvas_image(canvas)

    # Check shapes
    assert processed_28.shape == (28, 28)
    assert flat_784.shape == (1, 784)

    # Check value bounds [0.0, 1.0]
    assert processed_28.min() >= 0.0
    assert processed_28.max() <= 1.0
    assert processed_28.max() > 0.5  # Stroke should have significant intensity

    # Check flattening consistency
    np.testing.assert_array_equal(processed_28.reshape(1, 784), flat_784)


def test_preprocess_canvas_raises_on_empty():
    """Verifies EmptyCanvasError is raised when canvas is blank."""
    blank = np.zeros((280, 280, 4), dtype=np.uint8)
    with pytest.raises(EmptyCanvasError):
        preprocess_canvas_image(blank)
