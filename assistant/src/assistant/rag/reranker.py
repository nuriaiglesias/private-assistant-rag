from __future__ import annotations

from typing import List

from assistant.core.config import AppConfig
from assistant.rag.retriever import RetrievedChunk


class Reranker:
	"""
	Placeholder reranker.

	Current implementation preserves or sorts chunks by their retrieval score.
	This component is intentionally isolated so it can later be replaced by a
	cross-encoder, LLM-based reranker, or another reranking strategy.
	"""
	def __init__(self, config: AppConfig) -> None:
		self._config = config

	def rerank(
		self,
		query: str,
		chunks: List[RetrievedChunk],
		*,
		top_k: int = 5,
	) -> List[RetrievedChunk]:
		_ = query
		sorted_chunks = sorted(chunks, key=lambda item: item.score, reverse=True)
		return sorted_chunks[:top_k]
