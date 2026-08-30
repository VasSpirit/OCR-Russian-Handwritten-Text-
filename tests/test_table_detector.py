import numpy as np
from src.table_detector import TableDetector


def test_table_detector_uses_corrected_columns_and_25_rows():
    geom = TableDetector(dpi=300).detect(np.full((3508, 2480), 255, dtype=np.uint8))
    assert geom.n_rows == 25
    assert geom.column_bounds('row_no') == (36, 128)
    assert geom.column_bounds('street') == (128, 744)
    assert geom.column_bounds('house') == (744, 868)
