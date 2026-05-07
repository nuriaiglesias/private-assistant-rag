from __future__ import annotations

from assistant.core.config import AppConfig
from assistant.llm.base import LLMClient
from assistant.llm.ollama_provider import OllamaClient
from assistant.llm.openai_provider import OpenAICompatibleClient


def build_llm_client(config: AppConfig) -> LLMClient:
	provider = config.llm_provider.lower()
	if provider == "vllm":
		return OpenAICompatibleClient(
			base_url=config.llm_base_url,
			api_key=config.llm_api_key,
			model=config.llm_model,
		)
	if provider == "ollama":
		return OllamaClient(
			base_url=config.llm_base_url,
			model=config.llm_model,
		)

	raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
