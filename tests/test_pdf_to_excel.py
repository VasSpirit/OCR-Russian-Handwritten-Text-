import os
from pathlib import Path

import pytest

from src.pipeline import Pipeline


@pytest.mark.integration
def test_real_scan_pdf_to_excel_when_available(tmp_path):
    candidates = [Path('Скан_20260828.pdf'), Path('/workspace/Скан_20260828.pdf')]
    pdf = next((p for p in candidates if p.exists()), None)
    if pdf is None:
        pytest.skip('Скан_20260828.pdf is not available in this environment')
    out = tmp_path / 'result.xlsx'
    result = Pipeline(output_dir=tmp_path / 'output').process_to_excel(pdf, out)
    assert out.exists()
    assert result.summary.rows == 25
