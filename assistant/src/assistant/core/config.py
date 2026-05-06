from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppConfig:
	qdrant_url: str
	qdrant_collection: str
	embedding_model: str
	chunk_size: int
	chunk_overlap: int


def load_config() -> AppConfig:
	return AppConfig(
		qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
		qdrant_collection=os.getenv("QDRANT_COLLECTION", "assistant_docs"),
		embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
		chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
		chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
	)
