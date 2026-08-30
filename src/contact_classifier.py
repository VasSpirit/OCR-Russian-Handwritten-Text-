"""OpenCV heuristic classifier for + / 0 / - contact result cells."""
from __future__ import annotations

from enum import Enum

import cv2
import numpy as np

from src.models import RecognitionResult


class ContactClass(str, Enum):
    EMPTY = "EMPTY"
    PLUS = "PLUS"
    ZERO = "ZERO"
    MINUS = "MINUS"
    OTHER = "OTHER"


class ContactClassifier:
    def classify(self, image: np.ndarray) -> RecognitionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if gray.size == 0:
            return RecognitionResult(text=ContactClass.EMPTY.value, confidence=0.0, needs_review=True)
        bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        ink_ratio = float((bw > 0).sum()) / bw.size
        if ink_ratio < 0.01:
            return RecognitionResult(text=ContactClass.EMPTY.value, raw="", confidence=0.95, needs_review=False)
        lines = cv2.HoughLinesP(bw, 1, np.pi / 180, threshold=20, minLineLength=max(8, gray.shape[1] // 6), maxLineGap=5)
        angles: list[float] = []
        if lines is not None:
            for line in lines.reshape(-1, 4):
                x1, y1, x2, y2 = line
                angles.append(abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))))
        has_horizontal = any(a < 20 or a > 160 for a in angles)
        has_vertical = any(70 < a < 110 for a in angles)
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circular = False
        for c in contours:
            area = cv2.contourArea(c)
            if area < 20:
                continue
            peri = cv2.arcLength(c, True)
            if peri and 4 * np.pi * area / (peri * peri) > 0.45:
                circular = True
        if has_horizontal and has_vertical:
            return RecognitionResult(text=ContactClass.PLUS.value, raw="+", confidence=0.80, needs_review=False)
        if circular:
            return RecognitionResult(text=ContactClass.ZERO.value, raw="0", confidence=0.70, needs_review=False)
        if has_horizontal:
            return RecognitionResult(text=ContactClass.MINUS.value, raw="-", confidence=0.75, needs_review=False)
        return RecognitionResult(text=ContactClass.OTHER.value, raw="", confidence=0.30, needs_review=True)
