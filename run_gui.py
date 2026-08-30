#!/usr/bin/env python3
"""Tkinter GUI for running the production OCR pipeline."""
from __future__ import annotations

import logging
import queue
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.pipeline import Pipeline, PipelineResult

LOG_PATH = Path("output/logs/app.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=LOG_PATH, level=logging.ERROR, format="%(asctime)s %(levelname)s %(message)s")


class OcrGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("OCR PDF → Excel")
        self.geometry("560x260")
        self.input_pdf: Path | None = None
        self.result: PipelineResult | None = None
        self.events: queue.Queue[tuple] = queue.Queue()
        self._build_widgets()
        self.after(100, self._poll_events)

    def _build_widgets(self) -> None:
        self.pdf_label = ttk.Label(self, text="PDF не выбран")
        self.pdf_label.pack(fill="x", padx=16, pady=(16, 8))
        ttk.Button(self, text="Выбрать PDF", command=self._select_pdf).pack(padx=16, pady=4)
        self.recognize_button = ttk.Button(self, text="Распознать", command=self._start_recognition, state="disabled")
        self.recognize_button.pack(padx=16, pady=4)
        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", padx=16, pady=12)
        self.status = ttk.Label(self, text="Готово")
        self.status.pack(fill="x", padx=16, pady=4)
        self.save_button = ttk.Button(self, text="Сохранить Excel", command=self._save_excel, state="disabled")
        self.save_button.pack(padx=16, pady=12)

    def _select_pdf(self) -> None:
        filename = filedialog.askopenfilename(title="Выберите PDF", filetypes=[("PDF files", "*.pdf")])
        if filename:
            self.input_pdf = Path(filename)
            self.pdf_label.config(text=str(self.input_pdf))
            self.recognize_button.config(state="normal")
            self.save_button.config(state="disabled")
            self.result = None

    def _start_recognition(self) -> None:
        if not self.input_pdf:
            return
        self.recognize_button.config(state="disabled")
        self.save_button.config(state="disabled")
        self.progress.config(value=0, maximum=1)
        self.status.config(text="Распознавание...")
        threading.Thread(target=self._worker, args=(self.input_pdf,), daemon=True).start()

    def _worker(self, pdf_path: Path) -> None:
        try:
            def progress(done: int, total: int) -> None:
                self.events.put(("progress", done, total))
            result = Pipeline().process_pdf(pdf_path, progress_callback=progress)
            self.events.put(("done", result))
        except Exception as exc:  # GUI boundary: log full traceback and show a dialog on main thread.
            logging.error("OCR failed\n%s", traceback.format_exc())
            self.events.put(("error", exc))

    def _poll_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "progress":
                _, done, total = event
                self.progress.config(maximum=total, value=done)
                self.status.config(text=f"Обработано страниц: {done}/{total}")
            elif kind == "done":
                self.result = event[1]
                s = self.result.summary
                self.status.config(text=f"Готово: страниц {s.pages}, строк {s.rows}, полей с низкой уверенностью {s.low_confidence_fields}")
                self.save_button.config(state="normal")
                self.recognize_button.config(state="normal")
            elif kind == "error":
                self.recognize_button.config(state="normal")
                self.status.config(text="Ошибка распознавания")
                messagebox.showerror("Ошибка", f"Не удалось распознать PDF. Подробности в {LOG_PATH}\n\n{event[1]}")
        self.after(100, self._poll_events)

    def _save_excel(self) -> None:
        if not self.result:
            return
        filename = filedialog.asksaveasfilename(title="Сохранить Excel", defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if filename:
            try:
                self.result.to_excel(filename)
                messagebox.showinfo("Готово", f"Файл сохранён: {filename}")
            except Exception as exc:
                logging.error("Excel save failed\n%s", traceback.format_exc())
                messagebox.showerror("Ошибка", f"Не удалось сохранить Excel. Подробности в {LOG_PATH}\n\n{exc}")


if __name__ == "__main__":
    OcrGui().mainloop()
