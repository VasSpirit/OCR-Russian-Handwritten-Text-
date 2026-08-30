"""Confidence helpers."""
from __future__ import annotations


def needs_review(confidence: float, threshold: float = 0.70, valid: bool = True) -> bool:
    return (confidence < threshold) or not valid
