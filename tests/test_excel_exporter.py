import pandas as pd
from src.excel_exporter import ExcelExporter
from src.models import RecognitionDocument, RecognitionField, RecognitionRow


def test_excel_exporter_writes_all_sheets(tmp_path):
    row = RecognitionRow(page=1, row_number=1, fields={'street': RecognitionField(column='street', corrected_text='Дзержинского', confidence=0.9, needs_review=False)})
    doc = RecognitionDocument(pages=1, rows=[row], statistics={'pages': 1, 'rows': 1})
    out = tmp_path / 'result.xlsx'
    ExcelExporter().export(doc, out)
    xls = pd.ExcelFile(out)
    assert set(xls.sheet_names) == {'Результат', 'Ошибки', 'Raw OCR', 'Статистика'}
    assert pd.read_excel(out, sheet_name='Результат').iloc[0]['Улица'] == 'Дзержинского'
