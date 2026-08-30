# Benchmark and HTR model notes

## Existing project findings

- `README.md` describes this repository as an exploratory baseline: PDF rendering and table geometry probes exist, while post-processing and Excel export were previously TODO.
- `experiments/data/cols.txt` contains detected vertical line positions. The important correction is that the street cell spans `128..744`; interpreting `36..128` as street places the row-number column into the street field.
- `experiments/data/rows.txt` contains row centers from earlier probes. For the real target document the production template uses the first 25 rows requested for `Скан_20260828.pdf`.
- `experiments/run_easyocr_probes.py` runs EasyOCR over `/tmp/probes/*.png`; it is a baseline experiment, not production HTR.

## Candidate HTR models checked

| Candidate | Source | License | CPU/GPU | Python | Fine-tuning | Notes |
|---|---|---|---|---|---|---|
| SHIFT OCR / `shiftlab_ocr` | https://github.com/konverner/shiftlab_ocr / PyPI | MIT | CPU expected; GPU depends on backend | pip package | allowed by MIT | Targets handwritten Cyrillic text. Selected as preferred open-source HTR direction, but package API needs stabilization in `HTRRecognizer`. |
| `Kansallisarkisto/cyrillic-large-handwritten` | Hugging Face | model card must be checked before commercial use | Transformers CPU/GPU | Python/Transformers | likely, subject to license | Cyrillic/Russian historical handwriting. Good candidate for a future TrOCR backend. |
| Microsoft TrOCR handwritten | Hugging Face / Transformers | model card license must be checked | CPU/GPU | Python/Transformers | yes, model-dependent | Strong generic HTR architecture, but base checkpoints are not Russian-specific. |
| EasyOCR | PyPI/GitHub | Apache-2.0 | CPU/GPU | Python | not the intended path | Kept only as baseline/fallback; project README already shows poor quality on this handwriting. |

## Benchmark results in this environment

Ground truth: unavailable except the visually verified note that page 1 street is `Дзержинского`.

Real `Скан_20260828.pdf` run: not executed in this container because the PDF file was not present under `/workspace` or the repository root during this run.

No accuracy value is reported because no full verified ground-truth table is available and fabricating one would invalidate the benchmark.
