"""Utilities for mapping raw model generations to canonical labels."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

__all__ = ["canonicalize_label", "infer_label", "infer_labels"]

_LABEL_PATTERN = re.compile(
    r"label\s*[:=-]\s*"
    r"(early(?:\s+active)?(?:\s+pregnancy)?"
    r"|late(?:\s+active)?(?:\s+pregnancy)?"
    r"|no\s+active\s+pregnancy"
    r"|unrelated(?:\s+or\s+no\s+current\s+pregnancy)?"
    r"|no\s+current\s+pregnancy)",
    re.IGNORECASE,
)

_CANONICAL = {
    "early": "early",
    "early pregnancy": "early",
    "early active": "early",
    "early active pregnancy": "early",
    "late": "late",
    "late pregnancy": "late",
    "late active": "late",
    "late active pregnancy": "late",
    "unrelated": "unrelated",
    "unrelated or no current pregnancy": "unrelated",
    "no current pregnancy": "unrelated",
    "no active pregnancy": "unrelated",
    "no active": "unrelated",
    "none": "unrelated",
    "no": "unrelated",
}


def canonicalize_label(value: object) -> Optional[str]:
    """Map supported label variants to canonical {early, late, unrelated}."""

    if value is None:
        return None

    text = str(value).strip().lower().rstrip(".")
    text = re.sub(r"\s+", " ", text)

    if text in _CANONICAL:
        return _CANONICAL[text]

    if text.startswith("early") and "pregnancy" in text:
        return "early"
    if text.startswith("late") and "pregnancy" in text:
        return "late"
    if "no active pregnancy" in text:
        return "unrelated"

    return None


def infer_label(text: str) -> str:
    """Infer the final label from a raw generation.

    The heuristic mirrors the notebook implementation used during
    experimentation: prefer the *last* explicit ``Label:`` mention when
    present, otherwise fall back to the last keyword occurrence.
    """

    if not isinstance(text, str):
        return "UNK"

    text_lower = text.lower()

    # Prefer explicit ``Label: <answer>`` mentions (keep the last one)
    matches = list(_LABEL_PATTERN.finditer(text_lower))
    if matches:
        label = matches[-1].group(1).strip()
        canon = canonicalize_label(label)
        return canon if canon is not None else "UNK"

    word_patterns = {
        "unrelated": [
            r"\bno\s+active\s+pregnancy\b",
            r"\bunrelated\b",
            r"\bno\s+current\s+pregnancy\b",
        ],
        "late": [r"\blate\s+active\s+pregnancy\b", r"\blate\s+pregnancy\b", r"\blate\b"],
        "early": [r"\bearly\s+active\s+pregnancy\b", r"\bearly\s+pregnancy\b", r"\bearly\b"],
    }

    positions: dict[str, int] = {}
    for label, patterns in word_patterns.items():
        best = -1
        for pattern in patterns:
            for match in re.finditer(pattern, text_lower):
                best = max(best, match.start())
        if best != -1:
            positions[label] = best

    if not positions:
        return "UNK"

    # Choose the label with the last mention (closest to the end of the text)
    return max(positions, key=positions.get)


def infer_labels(values: Iterable[str]) -> List[str]:
    """Vectorised convenience wrapper around :func:`infer_label`."""

    return [infer_label(value) for value in values]
