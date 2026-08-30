import pymupdf
from src.models import RecognitionResult
from src.pipeline import Pipeline


class StubHTR:
    engine_name = 'stub-htr'
    def recognize(self, image):
        return RecognitionResult(text='Дзержинского', raw='Дзержинского', confidence=0.95, needs_review=False)

class StubNumeric:
    def recognize(self, image, field):
        return RecognitionResult(text='1', raw='1', confidence=0.95, needs_review=False)

class StubContact:
    def classify(self, image):
        return RecognitionResult(text='EMPTY', raw='', confidence=0.95, needs_review=False)


def test_pipeline_processes_pdf_and_writes_debug_outputs(tmp_path):
    pdf = tmp_path / 'sample.pdf'
    doc = pymupdf.open(); page = doc.new_page(width=595, height=842); page.insert_text((72,72),'sample'); doc.save(pdf); doc.close()
    p = Pipeline(output_dir=tmp_path / 'out', dpi=72)
    p.htr = StubHTR(); p.numeric = StubNumeric(); p.contact = StubContact()
    result = p.process(pdf)
    assert result.summary.pages == 1
    assert result.summary.rows == 25
    assert (tmp_path / 'out/debug/page_001_grid.png').exists()
    assert (tmp_path / 'out/debug/page_001_cells.jpg').exists()
