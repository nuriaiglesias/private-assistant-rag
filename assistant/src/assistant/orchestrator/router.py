from __future__ import annotations

"""Question routing helpers for orchestrator intent and type classification."""

import unicodedata

from assistant.orchestrator.rules import (
    COMPARATIVE_MARKERS,
    EMAIL_MARKERS,
    MULTI_DOC_MARKERS,
    UNANSWERABLE_MARKERS,
)
from assistant.orchestrator.state import IntentType, QuestionType


def normalize_text(text: str) -> str:
    """Normalize text to ASCII lowercase for deterministic matching."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()


def detect_intent(question: str) -> IntentType:
    """Detect whether the question implies an email action or general information request."""
    q = normalize_text(question)
    if any(marker in q for marker in EMAIL_MARKERS):
        return "email_action"
    return "information"


def classify_question_type(question: str) -> QuestionType:
    """Classify the question as single_doc, multi_doc, comparative, or unanswerable_risk."""
    q = normalize_text(question)

    if any(marker in q for marker in UNANSWERABLE_MARKERS):
        return "unanswerable_risk"

    if any(marker in q for marker in COMPARATIVE_MARKERS):
        return "comparative"

    if any(marker in q for marker in MULTI_DOC_MARKERS):
        return "multi_doc"

    return "single_doc"
