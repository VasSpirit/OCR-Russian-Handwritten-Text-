#!/usr/bin/env python3
"""CLI entry point for the production OCR pipeline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from threading import Event

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import Pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Process a PDF with the production OCR pipeline.")
    parser.add_argument("--input", required=True, help="Input PDF path")
    parser.add_argument("--output", required=True, help="Output XLSX path")
    args = parser.parse_args()

    def progress(done: int, total: int, message: str) -> None:
        print(f"{done}/{total}: {message}", flush=True)

    result = Pipeline().process_to_excel(args.input, args.output, progress_callback=progress, cancel_event=Event())
    s = result.summary
    print(f"Done: pages={s.pages}, rows={s.rows}, fields={s.fields}, needs_review={s.needs_review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
