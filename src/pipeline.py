"""Production PDF OCR pipeline shared by GUI and CLI."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

import cv2
import numpy as np
import pymupdf

from src.cell_extractor import CellExtractor
from src.confidence import needs_review
from src.contact_classifier import ContactClassifier
from src.dictionary_corrector import DictionaryCorrector
from src.excel_exporter import ExcelExporter
from src.handwriting_ocr import HTRRecognizer
from src.image_preprocessing import preprocess_handwriting
from src.models import RecognitionDocument, RecognitionField, RecognitionResult, RecognitionRow, mean_confidence
from src.numeric_ocr import NumericOCR
from src.table_detector import TableDetector

ProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True)
class PipelineSummary:
    pages: int
    rows: int
    fields: int
    needs_review: int


@dataclass
class PipelineResult:
    document: RecognitionDocument

    @property
    def summary(self) -> PipelineSummary:
        return PipelineSummary(self.document.pages, self.document.row_count, self.document.fields_count, self.document.needs_review_count)

    def to_excel(self, output_path: str | Path) -> None:
        ExcelExporter().export(self.document, output_path)


class PipelineCancelled(RuntimeError):
    pass


class Pipeline:
    """Reusable production pipeline: PDF -> cells -> field OCR/classifiers -> document."""

    def __init__(self, output_dir: str | Path = "output", dpi: int = 300) -> None:
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.table_detector = TableDetector(dpi=dpi)
        self.cell_extractor = CellExtractor(self.output_dir)
        self.htr = HTRRecognizer()
        self.numeric = NumericOCR()
        self.contact = ContactClassifier()
        self.corrector = DictionaryCorrector()

    def process(self, pdf_path: str | Path, progress_callback: ProgressCallback | None = None,
                cancel_event: Event | None = None) -> PipelineResult:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        rows: list[RecognitionRow] = []
        with pymupdf.open(pdf_path) as doc:
            total = len(doc)
            for page_number, page in enumerate(doc, start=1):
                self._raise_if_cancelled(cancel_event)
                if progress_callback:
                    progress_callback(page_number - 1, total, f"Обработка страницы {page_number} из {total}...")
                image = self._render_page(page)
                geometry = self.table_detector.detect(image)
                self.cell_extractor.save_grid(self.table_detector.draw_debug_grid(image, geometry), page_number)
                cells = self.cell_extractor.extract_page(image, geometry, page_number)
                self.cell_extractor.save_contact_sheet(cells, page_number, rows=min(25, geometry.n_rows))
                rows.extend(self._recognize_cells(cells, cancel_event))
                if progress_callback:
                    progress_callback(page_number, total, f"Страница {page_number} из {total} обработана")
        document = RecognitionDocument(pages=total if 'total' in locals() else 0, rows=rows)
        document.statistics = {
            "pages": document.pages,
            "rows": document.row_count,
            "fields": document.fields_count,
            "needs_review": document.needs_review_count,
            "average_confidence": mean_confidence([row.confidence for row in rows]),
            "htr_engine": self.htr.engine_name,
        }
        return PipelineResult(document)

    def process_to_excel(self, input_pdf: str | Path, output_xlsx: str | Path,
                         progress_callback: ProgressCallback | None = None,
                         cancel_event: Event | None = None) -> PipelineResult:
        result = self.process(input_pdf, progress_callback=progress_callback, cancel_event=cancel_event)
        result.to_excel(output_xlsx)
        return result

    def _recognize_cells(self, cells, cancel_event: Event | None) -> list[RecognitionRow]:
        grouped: dict[tuple[int, int], list] = {}
        for cell in cells:
            grouped.setdefault((cell.page, cell.row), []).append(cell)
        rows: list[RecognitionRow] = []
        for (page, row_no), row_cells in sorted(grouped.items()):
            self._raise_if_cancelled(cancel_event)
            row = RecognitionRow(page=page, row_number=row_no)
            for cell in row_cells:
                row.fields[cell.column] = self._recognize_cell(cell)
            rows.append(row)
        return rows

    def _recognize_cell(self, cell) -> RecognitionField:
        image = cv2.imread(str(cell.image_path), cv2.IMREAD_GRAYSCALE)
        prepared = preprocess_handwriting(image)
        if cell.column in {"house", "building", "apartment", "row_no"}:
            result = self.numeric.recognize(prepared, "house" if cell.column == "row_no" else cell.column)
            correction = self.corrector.correct(result.text)
        elif cell.column == "contact_result":
            result = self.contact.classify(prepared)
            correction = self.corrector.correct(result.text, "contact_results")
        else:
            result = self.htr.recognize(prepared)
            correction = self.corrector.correct(result.text, "streets" if cell.column == "street" else None)
        valid = bool(correction.corrected_text) or cell.column in {"comment", "check", "contact_result"}
        review = needs_review(result.confidence, valid=valid) or result.needs_review
        return RecognitionField(
            column=cell.column,
            raw_text=result.raw or result.text,
            normalized_text=correction.normalized_text,
            corrected_text=correction.corrected_text,
            confidence=result.confidence,
            needs_review=review,
            correction_method=correction.correction_method,
            correction_confidence=correction.correction_confidence,
            source_of_correction=correction.source_of_correction,
            reason="low_confidence_or_validation" if review else "",
            image_path=str(cell.image_path),
        )

    def _render_page(self, page: pymupdf.Page) -> np.ndarray:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(self.dpi / 72, self.dpi / 72), alpha=False)
        image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if pix.n == 3 else image

    @staticmethod
    def _raise_if_cancelled(cancel_event: Event | None) -> None:
        if cancel_event and cancel_event.is_set():
            raise PipelineCancelled("Processing cancelled")
