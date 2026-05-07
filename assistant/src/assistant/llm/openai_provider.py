from __future__ import annotations

from typing import List

import requests

from assistant.llm.base import ChatMessage, LLMClient


class OpenAICompatibleClient(LLMClient):
	def __init__(self, base_url: str, api_key: str, model: str) -> None:
		self._base_url = base_url.rstrip("/")
		self._api_key = api_key
		self._model = model

	def generate(self, messages: List[ChatMessage], temperature: float, max_tokens: int) -> str:
		payload = {
			"model": self._model,
			"messages": [message.__dict__ for message in messages],
			"temperature": temperature,
			"max_tokens": max_tokens,
		}
		headers = {"Content-Type": "application/json"}
		if self._api_key:
			headers["Authorization"] = f"Bearer {self._api_key}"

		response = requests.post(
			f"{self._base_url}/v1/chat/completions",
			headers=headers,
			json=payload,
			timeout=120,
		)
		response.raise_for_status()
		data = response.json()
		return data["choices"][0]["message"]["content"].strip()
