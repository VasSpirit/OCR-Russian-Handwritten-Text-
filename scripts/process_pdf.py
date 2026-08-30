#!/usr/bin/env python3
"""CLI entry point for the production OCR pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import Pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Process a PDF with the production OCR pipeline.")
    parser.add_argument("--input", required=True, help="Input PDF path")
    parser.add_argument("--output", required=True, help="Output XLSX path")
    args = parser.parse_args()

    def progress(done: int, total: int) -> None:
        print(f"Processed page {done}/{total}", flush=True)

    result = Pipeline().process_to_excel(args.input, args.output, progress_callback=progress)
    print(
        f"Done: pages={result.summary.pages}, rows={result.summary.rows}, "
        f"low_confidence_fields={result.summary.low_confidence_fields}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
