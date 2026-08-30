"""Excel exporter for recognition documents."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models import RecognitionDocument, RecognitionRow


class ExcelExporter:
    result_columns = ["№", "Страница", "Строка", "Улица", "Дом", "Корпус", "Квартира", "Пол", "Возраст", "Результат контакта", "Примечание", "Confidence", "Нужна проверка"]
    error_columns = ["Страница", "Строка", "Поле", "Raw OCR", "Исправленное значение", "Confidence", "Причина"]
    raw_columns = ["page", "row", "column", "raw_text", "normalized_text", "corrected_text", "confidence"]

    def export(self, document: RecognitionDocument, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            pd.DataFrame([self._result_row(r) for r in document.rows], columns=self.result_columns).to_excel(writer, index=False, sheet_name="Результат")
            pd.DataFrame(self._error_rows(document), columns=self.error_columns).to_excel(writer, index=False, sheet_name="Ошибки")
            pd.DataFrame(self._raw_rows(document), columns=self.raw_columns).to_excel(writer, index=False, sheet_name="Raw OCR")
            pd.DataFrame([document.statistics]).to_excel(writer, index=False, sheet_name="Статистика")

    def _result_row(self, row: RecognitionRow) -> dict:
        gender, age = self._split_gender_age(row.get_text("gender_age"))
        return {
            "№": row.get_text("row_no") or row.row_number,
            "Страница": row.page,
            "Строка": row.row_number,
            "Улица": row.get_text("street"),
            "Дом": row.get_text("house"),
            "Корпус": row.get_text("building"),
            "Квартира": row.get_text("apartment"),
            "Пол": gender,
            "Возраст": age,
            "Результат контакта": row.get_text("contact_result"),
            "Примечание": row.get_text("comment"),
            "Confidence": row.confidence,
            "Нужна проверка": row.needs_review,
        }

    def _error_rows(self, document: RecognitionDocument) -> list[dict]:
        rows = []
        for row in document.rows:
            for field in row.fields.values():
                if field.needs_review:
                    rows.append({
                        "Страница": row.page,
                        "Строка": row.row_number,
                        "Поле": field.column,
                        "Raw OCR": field.raw_text,
                        "Исправленное значение": field.corrected_text,
                        "Confidence": field.confidence,
                        "Причина": field.reason or "low_confidence_or_validation",
                    })
        return rows

    def _raw_rows(self, document: RecognitionDocument) -> list[dict]:
        return [
            {
                "page": row.page,
                "row": row.row_number,
                "column": field.column,
                "raw_text": field.raw_text,
                "normalized_text": field.normalized_text,
                "corrected_text": field.corrected_text,
                "confidence": field.confidence,
            }
            for row in document.rows for field in row.fields.values()
        ]

    @staticmethod
    def _split_gender_age(value: str) -> tuple[str, str]:
        parts = value.split()
        gender = next((p for p in parts if p.upper() in {"М", "Ж", "M"}), "")
        age = next((p for p in parts if p.isdigit()), "")
        return gender, age
