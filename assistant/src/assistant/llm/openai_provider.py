from __future__ import annotations

"""OpenAI-compatible LLM client adapter for assistant responses."""

from typing import List

import requests

from assistant.llm.base import ChatMessage, LLMClient, LLMResponse


class OpenAICompatibleClient(LLMClient):
    """Adapter for OpenAI-compatible chat completion APIs."""
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model

    def generate(self, messages: List[ChatMessage], temperature: float, max_tokens: int) -> LLMResponse:
        tokens_key = (
            "max_completion_tokens"
            if self._base_url.startswith("https://api.openai.com")
            else "max_tokens"
        )
        payload = {
            "model": self._model,
            "messages": [message.__dict__ for message in messages],
            "temperature": temperature,
            tokens_key: max_tokens,
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
        usage = data.get("usage", {})
        return LLMResponse(
            content=data["choices"][0]["message"]["content"].strip(),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )
