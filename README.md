# OCR Russian Handwritten Text - table form recognition

Exploratory repo: pipeline for extracting a handwritten Russian table from a scanned
PDF(a single 10334x14600 px sheet at  300 DPI,  27 table rows)and recognizing
cell fields with local OCR/HTR models.

## Current status - honest baseline

Verified on the real sample: /workspace/Ckan_20260828.pdf. The PDF
contains ONE page ->a  10334x14600 px scan with a  27-row form table.



| Stage | Status |
|---|---|
| PDF -> page image | PyMuPDF at  300 DPI |
| Deskew | scan is already straight(no rotation found) |
| Table line detection | OpenCV projection profiling ->  8 columns,  27 rows |
| Cell crop | `experiments/gen_probes.py` |
| Handwriting OCR | EasyOCR(ru/en, CPU) is NOT usable for this handwriting - it outputs low-confidence gibberish on real cells |
| Numeric OCR | same verdict(see probes) |
| Post-processing | TODO |
| Excel export | TODO |



A specialized Russian handwritten-text HTR model(e.g. TrOCR or an attention CNN-GRU
fine-tuned on Russian handwriting, or SHIFT OCR)is still needed for the actual
cell-recognition stage. This repo keeps the table geometry probes and the reproduction
scripts so that an HTR engine can be swapped in later.



## Reproduce the probes



pip install pymupdf opencv-python-headless numpy easyocr pillow
python experiments/gen_probes.py            # writes /tmp/probes/*.png
python experiments/make_probes_montage.py   # writes /tmp/probes_montage.png
python experiments/run_easyocr_probes.py    # shows why EasyOCR fails here



Table geometry( detected with OpenCV projections at the 300-DPI page scale):


- Columns(x-bounds:
  row counter      [0,36]
  street name      [36,128]
  house number     [744,868]
  building block   [872,1008]
  apartment        [1012,1112]
  gender/age      [1116,1308]
  contact result   [1312,1544]
  remark           [1544,2068]
  check column     [2068,2372]
-  27 rows with centers spaced ~110 px apart



## Project layout


experiments/          probe scripts(geometry + EasyOCR baseline)
config/               template/config for a future HTR engine
data/dictionaries/   vocab files for fuzzy correction(to be filled from the form)



## Roadmap


1. Install an fine-tune a Russian handwriting HTR model(CPU-inference capable
2. Bind it behind `HandwritingRecognizer.recognize(image) -> {text, confidence}`
3. Cell classification for `+ / 0 / -` contact results(OpenCV features first, ML later
4. Dictionary/fuzzy correction with RapidFuzz
5. Confidence + `needs_review` flagging
6. Excel export(`#`, page, row, street, house, building block, apartment,
   gender, age, contact result, remark, confidence, needs review)
7. Streamlit review UI(accept/edit/skip, corrections.csv - training data for fine-tuning


## Production pipeline and GUI

Production code is kept outside `experiments/`:

```bash
python scripts/process_pdf.py --input Скан_20260828.pdf --output output/result.xlsx
python run_gui.py
```

The production pipeline writes visual geometry artifacts for review:

- `output/debug/page_001_grid.png` — table border, column lines, row lines and labels.
- `output/debug/page_001_cells.jpg` — contact sheet for the first rows/important columns.
- `output/cells/page_001/row_001/*.png` — individual cell crops.

Important geometry note: `experiments/data/cols.txt` stores detected table-line positions. The production extractor treats `36..128` as `row_no` and `128..744` as `street`; using `36..128` for `street` shifts row numbers into the street field and produces incorrect XLSX values.

EasyOCR remains only a baseline/fallback. The HTR abstraction is in `src/handwriting_ocr.py`; `HTRRecognizer` is the production-facing recognizer facade and can be replaced with a stronger Russian handwriting backend without changing GUI/CLI code.
