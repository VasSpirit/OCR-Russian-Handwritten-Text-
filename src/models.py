"""Shared data structures for the OCR pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TableGeometry:
    """Detected or template-based table layout."""
    width_px: int
    height_px: int
    dpi: int = 300
    columns: list[dict] = field(default_factory=list)   # [{name, x0, x1}]
    row_centers: list[int] = field(default_factory=list)
    row_height_px: int = 110

    @property
    def n_rows(self) -> int:
        return len(self.row_centers)

    @property
    def n_cols(self) -> int:
        return len(self.columns)

    def column_bounds(self, name: str) -> tuple:
        for col in self.columns:
            if col["name"] == name:
                return int(col["x0"]), int(col["x1"])
        raise KeyError(name)

    def row_bbox(self, row_index: int) -> tuple:
        yc = self.row_centers[row_index]
        half = self.row_height_px //  2
        return yc - half, yc + half


@dataclass
class Cell:
    """A single table cell crop."""
    page: int
    row: int            # 1-based row index
    column: str
    bbox: tuple   # x0,y0,x1,y1
    image: Any            # np.ndarray (gray/color)
    row_no: Optional[int] = None   # printed row counter from the form

    @property
    def name(self) -> str:
        return f"page_{self.page:03d}_row_{self.row:03d}_{self.column}"


@dataclass
class RecognitionResult:
    """Result of one OCR/HTR call."""
    text: str = ""
    raw_text: str = ""
    confidence: float = 0.0
    needs_review: bool = True

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
        }


def mean_confidence(confs: list) -> float:
    if not confs:
        return 0.0
    return float(sum(confs) / len(confs))
