from __future__ import annotations

"""Lightweight LLM interface abstractions used by the assistant pipeline."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int


class LLMClient:
    def generate(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        raise NotImplementedError
