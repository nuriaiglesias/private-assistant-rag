from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
	qdrant_url: str
	qdrant_collection: str
	embedding_model: str
	chunk_size: int
	chunk_overlap: int
	llm_provider: str
	llm_base_url: str
	llm_model: str
	llm_api_key: str
	llm_temperature: float
	llm_max_tokens: int
	phoenix_enabled: bool
	phoenix_host: str
	phoenix_port: int
	phoenix_project: str


def load_config() -> AppConfig:
	load_dotenv()
	return AppConfig(
		qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
		qdrant_collection=os.getenv("QDRANT_COLLECTION", "assistant_docs"),
		embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
		chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
		chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
		llm_provider=os.getenv("LLM_PROVIDER", "vllm"),
		llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000"),
		llm_model=os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
		llm_api_key=os.getenv("LLM_API_KEY", ""),
		llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
		llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
		phoenix_enabled=os.getenv("PHOENIX_ENABLED", "false").lower() == "true",
		phoenix_host=os.getenv("PHOENIX_HOST", "localhost"),
		phoenix_port=int(os.getenv("PHOENIX_PORT", "4317")),
		phoenix_project=os.getenv("PHOENIX_PROJECT", "assistant-rag"),
	)
