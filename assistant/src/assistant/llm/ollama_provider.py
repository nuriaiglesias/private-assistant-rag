from __future__ import annotations

from typing import List

import requests

from assistant.llm.base import ChatMessage, LLMClient, LLMResponse


class OllamaClient(LLMClient):
	def __init__(self, base_url: str, model: str) -> None:
		self._base_url = base_url.rstrip("/")
		self._model = model

	def generate(self, messages: List[ChatMessage], temperature: float, max_tokens: int) -> LLMResponse:
		payload = {
			"model": self._model,
			"messages": [message.__dict__ for message in messages],
			"stream": False,
			"options": {
				"temperature": temperature,
				"num_predict": max_tokens,
			},
		}
		response = requests.post(
			f"{self._base_url}/api/chat",
			json=payload,
			timeout=120,
		)
		response.raise_for_status()
		data = response.json()
		message = data.get("message", {})
		return LLMResponse(
			content=message.get("content", "").strip(),
			input_tokens=int(data.get("prompt_eval_count", 0)),
			output_tokens=int(data.get("eval_count", 0)),
		)
