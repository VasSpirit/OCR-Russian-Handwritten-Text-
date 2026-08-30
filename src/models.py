"""Shared data structures for the OCR pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class TableGeometry:
    """Detected or template-based table layout in rendered page pixels."""

    width_px: int
    height_px: int
    dpi: int = 300
    columns: list[dict] = field(default_factory=list)
    row_centers: list[int] = field(default_factory=list)
    row_height_px: int = 110
    table_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def n_rows(self) -> int:
        return len(self.row_centers)

    @property
    def n_cols(self) -> int:
        return len(self.columns)

    def column_bounds(self, name: str) -> tuple[int, int]:
        for col in self.columns:
            if col["name"] == name:
                return int(col["x0"]), int(col["x1"])
        raise KeyError(name)

    def row_bbox(self, row_index: int) -> tuple[int, int]:
        yc = self.row_centers[row_index]
        half = self.row_height_px // 2
        return yc - half, yc + half


@dataclass
class Cell:
    """A saved table cell crop."""

    page: int
    row: int
    column: str
    bbox: tuple[int, int, int, int]
    image_path: Path
    image: Any | None = None

    @property
    def name(self) -> str:
        return f"page_{self.page:03d}_row_{self.row:03d}_{self.column}"


@dataclass
class RecognitionResult:
    """Result of one OCR/HTR/classification call."""

    text: str = ""
    confidence: float = 0.0
    raw: str = ""
    needs_review: bool = True

    @property
    def raw_text(self) -> str:
        return self.raw

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "raw": self.raw,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
        }


@dataclass
class RecognitionField:
    column: str
    raw_text: str = ""
    normalized_text: str = ""
    corrected_text: str = ""
    confidence: float = 0.0
    needs_review: bool = True
    correction_method: str = "none"
    correction_confidence: float = 0.0
    source_of_correction: str = "none"
    reason: str = ""
    image_path: str = ""


@dataclass
class RecognitionRow:
    page: int
    row_number: int
    fields: dict[str, RecognitionField] = field(default_factory=dict)

    def get_text(self, column: str) -> str:
        field = self.fields.get(column)
        return field.corrected_text if field else ""

    @property
    def confidence(self) -> float:
        return mean_confidence([f.confidence for f in self.fields.values()])

    @property
    def needs_review(self) -> bool:
        return any(f.needs_review for f in self.fields.values())


@dataclass
class RecognitionDocument:
    pages: int
    rows: list[RecognitionRow] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def fields_count(self) -> int:
        return sum(len(row.fields) for row in self.rows)

    @property
    def needs_review_count(self) -> int:
        return sum(1 for row in self.rows for field in row.fields.values() if field.needs_review)


def mean_confidence(confs: list[float]) -> float:
    if not confs:
        return 0.0
    return float(sum(confs) / len(confs))
