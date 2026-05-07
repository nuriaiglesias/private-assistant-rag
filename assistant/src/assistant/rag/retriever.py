from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sentence_transformers import SentenceTransformer

from assistant.core.config import AppConfig
from assistant.rag.vector_store import QdrantVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
	text: str
	source: str
	chunk_index: int
	score: float


class Retriever:
	def __init__(self, config: AppConfig) -> None:
		self._config = config
		self._model = SentenceTransformer(config.embedding_model)
		self._store: QdrantVectorStore | None = None

	def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
		query_vector = self._model.encode([query])[0].tolist()
		store = self._get_store(vector_size=len(query_vector))
		results = store.search(query_vector, limit=top_k)

		chunks: List[RetrievedChunk] = []
		for result in results:
			payload = result.payload or {}
			chunks.append(
				RetrievedChunk(
					text=payload.get("text", ""),
					source=payload.get("source", "unknown"),
					chunk_index=int(payload.get("chunk_index", 0)),
					score=float(result.score),
				)
			)
		return chunks

	def _get_store(self, vector_size: int) -> QdrantVectorStore:
		if self._store is None:
			self._store = QdrantVectorStore(
				url=self._config.qdrant_url,
				collection=self._config.qdrant_collection,
				vector_size=vector_size,
			)
		return self._store
