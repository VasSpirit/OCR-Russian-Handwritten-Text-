#!/usr/bin/env python3
"""Tkinter GUI for the production OCR pipeline."""
from __future__ import annotations

import csv
import logging
import queue
import shutil
import threading
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from src.models import RecognitionField
from src.pipeline import Pipeline, PipelineCancelled, PipelineResult

LOG_PATH = Path("output/logs/app.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=LOG_PATH, level=logging.ERROR, format="%(asctime)s %(levelname)s %(message)s")


class OcrGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("OCR русских рукописных анкет")
        self.geometry("1100x680")
        self.input_pdf: Path | None = None
        self.result: PipelineResult | None = None
        self.events: queue.Queue[tuple] = queue.Queue()
        self.cancel_event = threading.Event()
        self._review_images: list[ImageTk.PhotoImage] = []
        self._build_widgets()
        self.after(100, self._poll_events)

    def _build_widgets(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x", padx=12, pady=10)
        ttk.Label(top, text="PDF:").pack(side="left")
        self.pdf_label = ttk.Label(top, text="не выбран")
        self.pdf_label.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(top, text="Выбрать PDF...", command=self._select_pdf).pack(side="left", padx=4)
        self.recognize_button = ttk.Button(top, text="Распознать", command=self._start_recognition, state="disabled")
        self.recognize_button.pack(side="left", padx=4)
        self.cancel_button = ttk.Button(top, text="Отмена", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=4)
        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", padx=12, pady=6)
        self.status = ttk.Label(self, text="Готово")
        self.status.pack(fill="x", padx=12, pady=4)
        cols = ("num", "page", "street", "house", "building", "apartment", "gender_age", "contact", "confidence")
        self.preview = ttk.Treeview(self, columns=cols, show="headings", height=22)
        headings = ["№", "Страница", "Улица", "Дом", "Корпус", "Квартира", "Пол/Возраст", "Результат", "Confidence"]
        widths = [55, 80, 220, 80, 80, 90, 110, 110, 90]
        for col, heading, width in zip(cols, headings, widths):
            self.preview.heading(col, text=heading)
            self.preview.column(col, width=width, anchor="w")
        self.preview.pack(fill="both", expand=True, padx=12, pady=8)
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=12, pady=10)
        self.review_button = ttk.Button(bottom, text="Требуют проверки", command=self._open_review, state="disabled")
        self.review_button.pack(side="left", padx=4)
        self.save_button = ttk.Button(bottom, text="Сохранить Excel...", command=self._save_excel, state="disabled")
        self.save_button.pack(side="right", padx=4)

    def _select_pdf(self) -> None:
        filename = filedialog.askopenfilename(title="Выберите PDF", filetypes=[("PDF files", "*.pdf")])
        if filename:
            self.input_pdf = Path(filename)
            self.pdf_label.config(text=str(self.input_pdf))
            self.recognize_button.config(state="normal")
            self.save_button.config(state="disabled")
            self.review_button.config(state="disabled")
            self.result = None
            self._clear_preview()

    def _start_recognition(self) -> None:
        if not self.input_pdf:
            return
        self.cancel_event.clear()
        self.recognize_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self.save_button.config(state="disabled")
        self.review_button.config(state="disabled")
        self.progress.config(value=0, maximum=1)
        self.status.config(text="Запуск OCR...")
        self._clear_preview()
        threading.Thread(target=self._worker, args=(self.input_pdf,), daemon=True).start()

    def _worker(self, pdf_path: Path) -> None:
        try:
            def progress(done: int, total: int, message: str) -> None:
                self.events.put(("progress", done, total, message))
            result = Pipeline().process(pdf_path, progress_callback=progress, cancel_event=self.cancel_event)
            self.events.put(("done", result))
        except PipelineCancelled as exc:
            self.events.put(("cancelled", exc))
        except Exception as exc:
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
                _, done, total, message = event
                self.progress.config(maximum=total, value=done)
                self.status.config(text=message)
            elif kind == "done":
                self.result = event[1]
                s = self.result.summary
                self.status.config(text=f"Страниц: {s.pages}; Строк: {s.rows}; Полей: {s.fields}; Требуют проверки: {s.needs_review}")
                self._fill_preview()
                self.save_button.config(state="normal")
                self.review_button.config(state="normal")
                self.recognize_button.config(state="normal")
                self.cancel_button.config(state="disabled")
            elif kind == "cancelled":
                self.status.config(text="Отменено")
                self.recognize_button.config(state="normal")
                self.cancel_button.config(state="disabled")
            elif kind == "error":
                self.recognize_button.config(state="normal")
                self.cancel_button.config(state="disabled")
                self.status.config(text="Ошибка распознавания")
                messagebox.showerror("Ошибка", f"Не удалось распознать PDF. Подробности в {LOG_PATH}\n\n{event[1]}")
        self.after(100, self._poll_events)

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.status.config(text="Отмена после текущей операции...")

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

    def _fill_preview(self) -> None:
        self._clear_preview()
        if not self.result:
            return
        for row in self.result.document.rows[:100]:
            self.preview.insert("", "end", values=(
                row.row_number, row.page, row.get_text("street"), row.get_text("house"),
                row.get_text("building"), row.get_text("apartment"), row.get_text("gender_age"),
                row.get_text("contact_result"), f"{row.confidence:.2f}",
            ))

    def _clear_preview(self) -> None:
        for item in self.preview.get_children():
            self.preview.delete(item)

    def _open_review(self) -> None:
        if not self.result:
            return
        win = tk.Toplevel(self)
        win.title("Поля, требующие проверки")
        win.geometry("900x620")
        canvas = tk.Canvas(win)
        frame = ttk.Frame(canvas)
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=frame, anchor="nw")
        self._review_images.clear()
        row_idx = 0
        for row in self.result.document.rows:
            for field in row.fields.values():
                if not field.needs_review:
                    continue
                self._add_review_row(frame, row.page, row.row_number, field, row_idx)
                row_idx += 1
        frame.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))

    def _add_review_row(self, parent: ttk.Frame, page: int, row_number: int, field: RecognitionField, grid_row: int) -> None:
        ttk.Label(parent, text=f"p{page} r{row_number} {field.column} conf={field.confidence:.2f}").grid(row=grid_row, column=0, sticky="w", padx=4, pady=4)
        entry = ttk.Entry(parent, width=35)
        entry.insert(0, field.corrected_text)
        entry.grid(row=grid_row, column=1, padx=4, pady=4)
        if field.image_path and Path(field.image_path).exists():
            img = Image.open(field.image_path).convert("RGB")
            img.thumbnail((180, 70))
            photo = ImageTk.PhotoImage(img)
            self._review_images.append(photo)
            ttk.Label(parent, image=photo).grid(row=grid_row, column=2, padx=4, pady=4)
        ttk.Button(parent, text="Сохранить", command=lambda: self._save_correction(page, row_number, field, entry.get())).grid(row=grid_row, column=3, padx=4, pady=4)

    def _save_correction(self, page: int, row_number: int, field: RecognitionField, value: str) -> None:
        corrections = Path("data/corrections.csv")
        new_file = not corrections.exists()
        corrections.parent.mkdir(parents=True, exist_ok=True)
        with corrections.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            if new_file:
                writer.writerow(["page", "row", "column", "old_value", "new_value", "image_path"])
            writer.writerow([page, row_number, field.column, field.corrected_text, value, field.image_path])
        if field.image_path and Path(field.image_path).exists() and value.strip():
            dst = Path("dataset/images") / f"page_{page:03d}_row_{row_number:03d}_{field.column}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(field.image_path, dst)
            labels = Path("dataset/labels.csv")
            if not labels.exists() or not labels.read_text(encoding="utf-8").strip():
                labels.write_text("image,text\n", encoding="utf-8")
            with labels.open("a", encoding="utf-8", newline="") as fh:
                csv.writer(fh).writerow([str(dst), value])
        field.corrected_text = value
        field.source_of_correction = "manual"
        field.needs_review = False
        messagebox.showinfo("Сохранено", "Исправление сохранено")


if __name__ == "__main__":
    OcrGui().mainloop()
