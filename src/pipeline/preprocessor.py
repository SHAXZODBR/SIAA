"""Image preprocessor for Sentinel Medical AI.

Normalizes, resizes, and prepares DICOM pixel arrays for model input.
Handles 8-bit, 12-bit, and 16-bit DICOM files uniformly.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional
from loguru import logger

from src.pipeline.dicom_loader import DicomStudy


def normalize_pixels(pixel_array: np.ndarray, bits_stored: int = 16) -> np.ndarray:
    """Normalize pixel values to [0.0, 1.0] float32 range.

    Handles different bit depths: 8-bit, 12-bit, 16-bit DICOM images.

    Args:
        pixel_array: Raw pixel array from DICOM.
        bits_stored: Bit depth of the DICOM image.

    Returns:
        Normalized float32 array in [0, 1] range.
    """
    arr = pixel_array.astype(np.float32)

    # Use actual min/max for robust normalization
    # (handles negative values from CT Hounsfield units)
    vmin = arr.min()
    vmax = arr.max()

    if vmax - vmin == 0:
        logger.warning("Constant pixel values — returning zeros")
        return np.zeros_like(arr, dtype=np.float32)

    normalized = (arr - vmin) / (vmax - vmin)
    return normalized


def apply_windowing(
    pixel_array: np.ndarray,
    window_center: Optional[float],
    window_width: Optional[float],
) -> np.ndarray:
    """Apply DICOM windowing (brightness/contrast) for optimal visualization.

    Args:
        pixel_array: Pixel array (can be any range).
        window_center: DICOM WindowCenter value.
        window_width: DICOM WindowWidth value.

    Returns:
        Windowed array in [0, 1] range.
    """
    if window_center is None or window_width is None:
        return normalize_pixels(pixel_array)

    lower = window_center - window_width / 2
    upper = window_center + window_width / 2

    arr = pixel_array.astype(np.float32)
    arr = np.clip(arr, lower, upper)
    arr = (arr - lower) / (upper - lower)
    return arr


def resize_with_padding(
    image: np.ndarray,
    target_size: int = 512,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Resize image to target_size x target_size preserving aspect ratio with padding.

    Args:
        image: 2D numpy array (H, W).
        target_size: Target dimension (square output).
        pad_value: Value for padding pixels.

    Returns:
        Resized and padded image of shape (target_size, target_size).
    """
    h, w = image.shape[:2]

    # Compute scale to fit within target_size
    scale = target_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded canvas
    canvas = np.full((target_size, target_size), pad_value, dtype=np.float32)

    # Center the resized image
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


def preprocess_study(
    study: DicomStudy,
    target_size: int = 512,
    use_windowing: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline for a single DICOM study.

    Steps:
    1. Apply windowing (if available)
    2. Normalize to [0, 1]
    3. Resize to target_size x target_size with padding

    Args:
        study: Loaded DicomStudy object.
        target_size: Output image size.
        use_windowing: Whether to apply DICOM windowing.

    Returns:
        Preprocessed image array of shape (target_size, target_size), float32, [0, 1].
    """
    pixel_array = study.pixel_array

    # Handle 3D arrays (multi-frame / RGB) — take first frame or convert
    if pixel_array.ndim == 3:
        if pixel_array.shape[2] == 3:
            # RGB — convert to grayscale
            pixel_array = np.mean(pixel_array, axis=2)
        else:
            # Multi-frame — take middle slice
            mid = pixel_array.shape[0] // 2
            pixel_array = pixel_array[mid]

    # Step 1: Windowing or normalize
    if use_windowing and study.window_center is not None:
        processed = apply_windowing(pixel_array, study.window_center, study.window_width)
    else:
        processed = normalize_pixels(pixel_array, study.bits_stored)

    # Step 2: Resize with padding
    processed = resize_with_padding(processed, target_size)

    logger.debug(
        f"Preprocessed: {study.file_path} → "
        f"{processed.shape}, range [{processed.min():.3f}, {processed.max():.3f}]"
    )
    return processed


def save_preprocessed(
    image: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save preprocessed image as PNG file.

    Args:
        image: Float32 array in [0, 1] range.
        output_path: Output PNG file path.

    Returns:
        Path to saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to 8-bit for PNG
    img_uint8 = (image * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), img_uint8)

    return output_path
