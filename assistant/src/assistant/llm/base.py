from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ChatMessage:
	role: str
	content: str


class LLMClient:
	def generate(self, messages: List[ChatMessage], temperature: float, max_tokens: int) -> str:
		raise NotImplementedError
