"""Conservative dictionary/fuzzy correction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from difflib import SequenceMatcher

try:
    from rapidfuzz import fuzz, process
except ImportError:  # optional until requirements are installed
    fuzz = None
    process = None


@dataclass
class CorrectionResult:
    raw_text: str
    normalized_text: str
    corrected_text: str
    correction_method: str = "none"
    correction_confidence: float = 0.0
    source_of_correction: str = "none"


class DictionaryCorrector:
    def __init__(self, dictionaries_dir: str | Path = "data/dictionaries", min_score: float = 92.0) -> None:
        self.dictionaries_dir = Path(dictionaries_dir)
        self.min_score = min_score
        self.entries = {name: self._load(name) for name in ["streets", "common_words", "contact_results", "names", "surnames"]}

    def correct(self, text: str, dictionary: str | None = None) -> CorrectionResult:
        normalized = self.normalize(text)
        if not normalized or not dictionary:
            return CorrectionResult(text, normalized, normalized)
        choices = self.entries.get(dictionary, [])
        if not choices:
            return CorrectionResult(text, normalized, normalized)
        if process is not None:
            match = process.extractOne(normalized, choices, scorer=fuzz.WRatio)
        else:
            scored = [(choice, SequenceMatcher(None, normalized.lower(), choice.lower()).ratio() * 100) for choice in choices]
            match = max(scored, key=lambda item: item[1]) if scored else None
        if match and match[1] >= self.min_score:
            method = "rapidfuzz" if process is not None else "difflib"
            return CorrectionResult(text, normalized, match[0], method, float(match[1]), "dictionary")
        return CorrectionResult(text, normalized, normalized)

    @staticmethod
    def normalize(text: str) -> str:
        text = text.replace("ё", "е").replace("Ё", "Е")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _load(self, name: str) -> list[str]:
        path = self.dictionaries_dir / f"{name}.txt"
        if not path.exists():
            return []
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]
