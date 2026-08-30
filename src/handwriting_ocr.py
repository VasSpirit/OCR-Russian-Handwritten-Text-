"""Replaceable handwritten text recognizers."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from src.models import RecognitionResult, mean_confidence


class HandwritingRecognizer(Protocol):
    def recognize(self, image: np.ndarray) -> RecognitionResult:
        ...


class EasyOCRRecognizer:
    """Baseline OCR adapter; retained for comparison and fallback."""

    def __init__(self, allowlist: str | None = None) -> None:
        import easyocr

        self.allowlist = allowlist
        self.reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        kwargs = {"paragraph": False}
        if self.allowlist:
            kwargs["allowlist"] = self.allowlist
        results = self.reader.readtext(image, **kwargs)
        parts = [r[1].strip() for r in results if r[1].strip()]
        conf = mean_confidence([float(r[2]) for r in results])
        raw = " | ".join(parts)
        return RecognitionResult(text=" ".join(parts), raw=raw, confidence=conf, needs_review=conf < 0.70 or not parts)


class HTRRecognizer:
    """Production HTR facade backed by SHIFT OCR when available.

    SHIFT OCR targets handwritten Cyrillic text and is MIT-licensed. We load its
    recognizer lazily so GUI startup stays responsive and so tests can replace the
    recognizer without downloading weights. If SHIFT OCR cannot be initialized, the
    facade falls back to the explicitly marked EasyOCR baseline.
    """

    def __init__(self) -> None:
        self.engine_name = "shiftlab_ocr"
        self._engine = None
        self._failed = False

    def _load_shiftlab(self):
        import os
        import urllib.request
        from pathlib import Path

        from shiftlab_ocr.doc2text.recognition import Recognizer

        weights = Path.home() / ".cache" / "ocr-russian-handwritten-text" / "ocr_transformer_4h2l_simple_conv_64x256.pt"
        weights.parent.mkdir(parents=True, exist_ok=True)
        if not weights.exists():
            urllib.request.urlretrieve(
                "https://github.com/konverner/shiftlab_ocr/raw/main/doc2text/weights/ocr_transformer_4h2l_simple_conv_64x256.pt",
                weights,
            )
        recognizer = Recognizer()
        recognizer.load_model(os.fspath(weights))
        return recognizer

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        if self._engine is None and not self._failed:
            try:
                self._engine = self._load_shiftlab()
            except Exception:
                self._failed = True
                self.engine_name = "easyocr-baseline-fallback"
                self._engine = EasyOCRRecognizer()
        if self.engine_name == "shiftlab_ocr":
            from PIL import Image

            pil_image = Image.fromarray(image).convert("RGB")
            text = self._engine.run(pil_image).strip()
            return RecognitionResult(text=text, raw=text, confidence=0.50 if text else 0.0, needs_review=not bool(text))
        return self._engine.recognize(image)
