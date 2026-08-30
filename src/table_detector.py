"""Table geometry detection for the known Russian handwritten form."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.models import TableGeometry


@dataclass(frozen=True)
class TemplateForm:
    width: int = 2480
    height: int = 3508
    row_height: int = 110
    # Corrected interpretation of experiments/data/cols.txt:
    # 36/40 and following pairs are table-line thicknesses; cell spans are line-to-line.
    columns: tuple[tuple[str, int, int], ...] = (
        ("row_no", 36, 128),
        ("street", 128, 744),
        ("house", 744, 868),
        ("building", 868, 1008),
        ("apartment", 1008, 1112),
        ("gender_age", 1112, 1308),
        ("contact_result", 1308, 1544),
        ("comment", 1544, 2068),
        ("check", 2068, 2372),
    )
    row_centers: tuple[int, ...] = (
        336, 446, 556, 666, 778, 888, 998, 1106, 1216, 1324,
        1436, 1546, 1658, 1770, 1882, 1994, 2102, 2214, 2326,
        2436, 2548, 2656, 2768, 2878, 2990,
    )


class TableDetector:
    """Detects/scales the table template and creates visual debug grids."""

    def __init__(self, dpi: int = 300, template: TemplateForm | None = None) -> None:
        self.dpi = dpi
        self.template = template or TemplateForm()

    def detect(self, page_image: np.ndarray) -> TableGeometry:
        height, width = page_image.shape[:2]
        sx = width / self.template.width
        sy = height / self.template.height
        columns = [
            {"name": name, "x0": round(x0 * sx), "x1": round(x1 * sx)}
            for name, x0, x1 in self.template.columns
        ]
        rows = [round(y * sy) for y in self.template.row_centers]
        x0 = min(col["x0"] for col in columns)
        x1 = max(col["x1"] for col in columns)
        y0 = max(0, rows[0] - round(self.template.row_height * sy / 2))
        y1 = min(height, rows[-1] + round(self.template.row_height * sy / 2))
        return TableGeometry(
            width_px=width,
            height_px=height,
            dpi=self.dpi,
            columns=columns,
            row_centers=rows,
            row_height_px=round(self.template.row_height * sy),
            table_bbox=(x0, y0, x1, y1),
        )

    def draw_debug_grid(self, page_image: np.ndarray, geometry: TableGeometry) -> np.ndarray:
        canvas = cv2.cvtColor(page_image, cv2.COLOR_GRAY2BGR) if page_image.ndim == 2 else page_image.copy()
        x0, y0, x1, y1 = geometry.table_bbox
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 255), 3)
        for col in geometry.columns:
            cx0, cx1 = int(col["x0"]), int(col["x1"])
            cv2.line(canvas, (cx0, y0), (cx0, y1), (255, 0, 0), 2)
            cv2.line(canvas, (cx1, y0), (cx1, y1), (255, 0, 0), 2)
            cv2.putText(canvas, col["name"], (cx0 + 4, max(20, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        for idx, yc in enumerate(geometry.row_centers, start=1):
            ry0, ry1 = geometry.row_bbox(idx - 1)
            cv2.line(canvas, (x0, ry0), (x1, ry0), (0, 128, 255), 2)
            cv2.line(canvas, (x0, ry1), (x1, ry1), (0, 128, 255), 1)
            cv2.putText(canvas, f"row={idx}", (max(0, x0 - 95), yc + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            for col in geometry.columns:
                if col["name"] in {"street", "house", "building", "apartment", "gender_age", "contact_result"}:
                    cv2.putText(canvas, f"r={idx} {col['name']}", (int(col["x0"]) + 3, yc), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 90, 0), 1)
        return canvas
