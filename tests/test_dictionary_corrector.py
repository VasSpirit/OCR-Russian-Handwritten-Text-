from src.dictionary_corrector import DictionaryCorrector


def test_dictionary_corrector_conservative_match(tmp_path):
    d = tmp_path / 'data'
    d.mkdir()
    (d / 'streets.txt').write_text('Дзержинского\n', encoding='utf-8')
    for name in ['common_words', 'contact_results', 'names', 'surnames']:
        (d / f'{name}.txt').write_text('', encoding='utf-8')
    result = DictionaryCorrector(d, min_score=80).correct('Дзержинсково', 'streets')
    assert result.corrected_text == 'Дзержинского'
    assert result.source_of_correction == 'dictionary'
