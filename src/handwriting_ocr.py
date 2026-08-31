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

    def __init__(self, allowlist: str | None = None, gpu: bool = False) -> None:
        import easyocr

        self.allowlist = allowlist
        self.reader = easyocr.Reader(["ru", "en"], gpu=gpu, verbose=False)

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
    recognizer lazily (``load_model``) so GUI startup stays responsive. If SHIFT OCR
    cannot be initialized, the facade falls back to the explicitly marked EasyOCR baseline.

    The model stays on ``device`` (``"cpu"`` or ``"cuda"``) and the EasyOCR fallback'
    receives the same device as its ``gpu`` flag.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.engine_name = "shiftlab_ocr"
        self.model_load_seconds: float = 0.0
        self._engine = None
        self._failed = False

    def load_model(self) -> None:
        """Load the HTR (or explicit fallback) model if not already loaded."""
        if self._engine is not None or self._failed:
            return
        import logging
        logger = logging.getLogger(__name__)
        logger.info("OCR_MODEL_LOADING engine=%s device=%s", self.engine_name, self.device)
        import time
        t0 = time.monotonic()
        try:
            self._engine = self._load_shiftlab()
        except Exception:
            logger.exception("OCR_MODEL_LOADING engine=shiftlab_ocr failed; falling back to easyocr")
            self._failed = True
            self.engine_name = "easyocr-baseline-fallback"
            self._engine = EasyOCRRecognizer(gpu=(self.device == "cuda"))
        self.model_load_seconds = time.monotonic() - t0
        logger.info("OCR_MODEL_READY engine=%s device=%s load_seconds=%.3f", self.engine_name, self.device, self.model_load_seconds)

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
        recognizer = Recognizer(device=torch_device(self.device))
        recognizer.load_model(os.fspath(weights), device=self.device)
        if self.device == "cuda":
            try:
                recognizer.model = recognizer.model.to(torch_device(self.device))
            except Exception:
                pass
        return recognizer

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        self.load_model()
        if self.engine_name == "shiftlab_ocr":
            from PIL import Image

            pil_image = Image.fromarray(image).convert("RGB")
            text = self._engine.run(pil_image).strip()
            return RecognitionResult(text=text, raw=text, confidence=0.50 if text else 0.0, needs_review=not bool(text))
        return self._engine.recognize(image)


def torch_device(device: str):
    import torch

    return torch.device(device)