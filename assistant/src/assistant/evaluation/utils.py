from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import List


def load_json(path: Path) -> List[dict]:
    """Load a JSON document and return its root array of objects."""
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(text: str) -> str:
    """Return a normalized lowercase version of the text without accents."""
    text = text.strip().lower()
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )
    return " ".join(text.split())


def tokenize_words(text: str, min_length: int = 4) -> List[str]:
    """Split text into normalized word tokens, filtering out short tokens."""
    return [word for word in normalize_text(text).split() if len(word) >= min_length]


REFUSAL_PATTERNS: List[str] = [
    "no dispongo de informacion",
    "no tengo informacion",
    "no hay informacion suficiente",
    "informacion insuficiente",
    "no puedo responder",
    "no se dispone de informacion",
]


def is_refusal(answer: str) -> bool:
    """Return True when the answer appears to be a refusal or insufficient-evidence response."""
    normalized = normalize_text(answer)
    return any(pattern in normalized for pattern in REFUSAL_PATTERNS)
