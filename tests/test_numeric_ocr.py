from src.models import RecognitionResult
from src.numeric_ocr import NumericOCR


class Stub:
    def __init__(self, text):
        self.text = text
    def recognize(self, image):
        return RecognitionResult(text=self.text, raw=self.text, confidence=0.91, needs_review=False)


def test_numeric_ocr_validates_digits():
    ocr = NumericOCR.__new__(NumericOCR)
    ocr.recognizer = Stub('12a')
    result = ocr.recognize(None, 'house')
    assert result.text == '12'
    assert result.needs_review is False


def test_numeric_ocr_flags_invalid_empty():
    ocr = NumericOCR.__new__(NumericOCR)
    ocr.recognizer = Stub('abc')
    result = ocr.recognize(None, 'apartment')
    assert result.needs_review is True
