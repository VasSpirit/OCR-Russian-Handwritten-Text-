"""Cell extraction and visual debug artifacts."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.models import Cell, TableGeometry


class CellExtractor:
    def __init__(self, output_dir: str | Path = "output") -> None:
        self.output_dir = Path(output_dir)

    def extract_page(self, page_image: np.ndarray, geometry: TableGeometry, page_number: int) -> list[Cell]:
        cells: list[Cell] = []
        for row_idx in range(geometry.n_rows):
            y0, y1 = geometry.row_bbox(row_idx)
            for col in geometry.columns:
                x0, x1 = int(col["x0"]), int(col["x1"])
                yy0, yy1 = max(0, y0), min(page_image.shape[0], y1)
                xx0, xx1 = max(0, x0), min(page_image.shape[1], x1)
                crop = page_image[yy0:yy1, xx0:xx1]
                path = self.output_dir / "cells" / f"page_{page_number:03d}" / f"row_{row_idx + 1:03d}" / f"{col['name']}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), crop)
                cells.append(Cell(page=page_number, row=row_idx + 1, column=col["name"], bbox=(xx0, yy0, xx1, yy1), image_path=path, image=crop))
        return cells

    def save_grid(self, grid_image: np.ndarray, page_number: int) -> Path:
        path = self.output_dir / "debug" / f"page_{page_number:03d}_grid.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), grid_image)
        return path

    def save_contact_sheet(self, cells: list[Cell], page_number: int, rows: int = 25) -> Path:
        wanted = ["street", "house", "building", "apartment", "gender_age", "contact_result"]
        by_key = {(c.row, c.column): c for c in cells}
        thumb_w, thumb_h, label_h = 220, 90, 24
        sheet = Image.new("RGB", (len(wanted) * thumb_w, rows * (thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()
        for row in range(1, rows + 1):
            for col_idx, column in enumerate(wanted):
                cell = by_key.get((row, column))
                x = col_idx * thumb_w
                y = (row - 1) * (thumb_h + label_h)
                if cell and cell.image_path.exists():
                    img = Image.open(cell.image_path).convert("RGB")
                    img.thumbnail((thumb_w - 6, thumb_h - 6))
                    sheet.paste(img, (x + 3, y + 3))
                draw.rectangle((x, y, x + thumb_w - 1, y + thumb_h + label_h - 1), outline="black")
                draw.text((x + 4, y + thumb_h + 4), f"row {row:02d} {column}", fill="red", font=font)
        path = self.output_dir / "debug" / f"page_{page_number:03d}_cells.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(path, quality=92)
        return path
