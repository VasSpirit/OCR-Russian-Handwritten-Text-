"""Image preprocessing for handwritten table-cell crops."""
from __future__ import annotations

import cv2
import numpy as np


def ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_table_borders(gray: np.ndarray) -> np.ndarray:
    """Remove horizontal/vertical ruling lines while preserving handwriting."""
    if gray.size == 0:
        return gray
    inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, gray.shape[1] // 3), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, gray.shape[0] // 2)))
    lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel)
    lines = cv2.bitwise_or(lines, cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel))
    return cv2.inpaint(gray, lines, 3, cv2.INPAINT_TELEA)


def preprocess_handwriting(image: np.ndarray, upscale: int = 2, padding: int = 12) -> np.ndarray:
    gray = ensure_gray(image)
    gray = remove_table_borders(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    if upscale > 1 and gray.size:
        gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(gray, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)
