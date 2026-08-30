"""Production PDF OCR pipeline shared by GUI and CLI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import cv2
import pymupdf
import numpy as np
import pandas as pd

from src.models import RecognitionResult, TableGeometry, mean_confidence

ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class PipelineSummary:
    pages: int
    rows: int
    low_confidence_fields: int


@dataclass
class PipelineResult:
    records: list[dict]
    summary: PipelineSummary

    def to_excel(self, output_path: str | Path) -> None:
        """Write recognition records to a real XLSX workbook."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.records)
        if df.empty:
            df = pd.DataFrame(columns=Pipeline.EXCEL_COLUMNS)
        else:
            df = df.reindex(columns=Pipeline.EXCEL_COLUMNS)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="OCR Results")


class EasyOcrRecognizer:
    """Thin adapter around EasyOCR. OCR details stay out of entry points."""

    def __init__(self) -> None:
        import easyocr

        self._reader = easyocr.Reader(["ru", "en"], gpu=False, verbose=False)

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        results = self._reader.readtext(image, paragraph=False)
        text_parts = [item[1].strip() for item in results if item[1].strip()]
        confidences = [float(item[2]) for item in results]
        raw_text = " | ".join(text_parts)
        confidence = mean_confidence(confidences)
        return RecognitionResult(
            text=" ".join(text_parts),
            raw_text=raw_text,
            confidence=confidence,
            needs_review=confidence < 0.70 or not text_parts,
        )


class Pipeline:
    """Reusable production pipeline for PDF table OCR and XLSX export."""

    EXCEL_COLUMNS = [
        "page", "row", "street", "house", "building_block", "apartment",
        "gender_age", "contact_result", "remark", "confidence", "needs_review",
        "low_confidence_fields",
    ]

    _TEMPLATE_WIDTH = 2480
    _TEMPLATE_HEIGHT = 3508
    _ROW_CENTERS = [
        336, 446, 556, 666, 778, 888, 998, 1106, 1216, 1324, 1436, 1546, 1658,
        1770, 1882, 1994, 2102, 2214, 2326, 2436, 2548, 2656, 2768, 2878, 2990, 3106,
    ]
    _COLUMNS = [
        ("street", 36, 128),
        ("house", 744, 868),
        ("building_block", 872, 1008),
        ("apartment", 1012, 1112),
        ("gender_age", 1116, 1308),
        ("contact_result", 1312, 1544),
        ("remark", 1544, 2068),
    ]

    def __init__(self, recognizer: EasyOcrRecognizer | None = None, dpi: int = 300) -> None:
        self.recognizer = recognizer or EasyOcrRecognizer()
        self.dpi = dpi

    def process_pdf(self, input_pdf: str | Path, progress_callback: ProgressCallback | None = None) -> PipelineResult:
        input_pdf = Path(input_pdf)
        if not input_pdf.exists():
            raise FileNotFoundError(f"PDF not found: {input_pdf}")

        records: list[dict] = []
        total_pages = 0
        with pymupdf.open(input_pdf) as doc:
            total_pages = len(doc)
            for page_index, page in enumerate(doc, start=1):
                image = self._render_page(page)
                geometry = self._geometry_for_image(image)
                records.extend(self._process_page(image, page_index, geometry))
                if progress_callback:
                    progress_callback(page_index, total_pages)

        low_fields = sum(int(record["low_confidence_fields"]) for record in records)
        return PipelineResult(records, PipelineSummary(total_pages, len(records), low_fields))

    def process_to_excel(self, input_pdf: str | Path, output_xlsx: str | Path,
                         progress_callback: ProgressCallback | None = None) -> PipelineResult:
        result = self.process_pdf(input_pdf, progress_callback=progress_callback)
        result.to_excel(output_xlsx)
        return result

    def _render_page(self, page: pymupdf.Page) -> np.ndarray:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(self.dpi / 72, self.dpi / 72), alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if pix.n == 3 else image

    def _geometry_for_image(self, image: np.ndarray) -> TableGeometry:
        height, width = image.shape[:2]
        sx = width / self._TEMPLATE_WIDTH
        sy = height / self._TEMPLATE_HEIGHT
        columns = [{"name": name, "x0": round(x0 * sx), "x1": round(x1 * sx)} for name, x0, x1 in self._COLUMNS]
        rows = [round(y * sy) for y in self._ROW_CENTERS]
        return TableGeometry(width_px=width, height_px=height, dpi=self.dpi, columns=columns, row_centers=rows, row_height_px=round(110 * sy))

    def _process_page(self, image: np.ndarray, page_number: int, geometry: TableGeometry) -> Iterable[dict]:
        for row_index in range(geometry.n_rows):
            row_data: dict[str, str | int | float | bool] = {"page": page_number, "row": row_index + 1}
            confidences: list[float] = []
            low_fields = 0
            for column in geometry.columns:
                x0, x1 = int(column["x0"]), int(column["x1"])
                y0, y1 = geometry.row_bbox(row_index)
                crop = image[max(0, y0):min(image.shape[0], y1), max(0, x0):min(image.shape[1], x1)]
                result = self.recognizer.recognize(self._prepare_crop(crop))
                row_data[column["name"]] = result.text
                confidences.append(result.confidence)
                low_fields += int(result.needs_review)
            row_data["confidence"] = mean_confidence(confidences)
            row_data["needs_review"] = low_fields > 0
            row_data["low_confidence_fields"] = low_fields
            yield row_data

    @staticmethod
    def _prepare_crop(crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return crop
        scale = 2 if min(crop.shape[:2]) < 80 else 1
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
