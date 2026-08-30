import numpy as np
from src.cell_extractor import CellExtractor
from src.table_detector import TableDetector


def test_cell_extractor_saves_named_cells(tmp_path):
    image = np.full((3508, 2480), 255, dtype=np.uint8)
    geom = TableDetector().detect(image)
    cells = CellExtractor(tmp_path).extract_page(image, geom, 1)
    assert len(cells) == 25 * 9
    assert (tmp_path / 'cells/page_001/row_001/street.png').exists()
    street = next(c for c in cells if c.row == 1 and c.column == 'street')
    assert street.bbox == (128, 281, 744, 391)
