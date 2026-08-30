"""Field-specific numeric OCR with validation."""
from __future__ import annotations

import re

import numpy as np

from src.handwriting_ocr import EasyOCRRecognizer
from src.models import RecognitionResult


class NumericOCR:
    patterns = {
        "house": re.compile(r"^[0-9]+$"),
        "building": re.compile(r"^[0-9]+$"),
        "apartment": re.compile(r"^[0-9]+$"),
        "age": re.compile(r"^[0-9]{1,3}$"),
    }

    def __init__(self) -> None:
        self.recognizer = None

    def recognize(self, image: np.ndarray, field: str) -> RecognitionResult:
        if self.recognizer is None:
            self.recognizer = EasyOCRRecognizer(allowlist="0123456789")
        result = self.recognizer.recognize(image)
        text = re.sub(r"\D+", "", result.text)
        valid = bool(self.patterns.get(field, self.patterns["house"]).match(text))
        return RecognitionResult(text=text, raw=result.raw, confidence=result.confidence, needs_review=result.confidence < 0.70 or not valid)
