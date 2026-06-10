from __future__ import annotations

from typing import List

from sentence_transformers import CrossEncoder

from assistant.core.config import AppConfig
from assistant.rag.retriever import RetrievedChunk


class Reranker:
    """
    Cross-encoder reranker.

    Scores each (query, chunk) pair and sorts by the cross-encoder score.
    """
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model: CrossEncoder | None = None

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        *,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._get_model().predict(pairs, batch_size=4)
        ranked = list(zip(chunks, scores))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return [
            RetrievedChunk(
                text=chunk.text,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=chunk.score,
                title=chunk.title,
                url=chunk.url,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                rerank_score=float(score),
            )
            for chunk, score in ranked[:top_k]
        ]

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            import torch
            self._model = CrossEncoder(
                self._config.reranker_model,
                model_kwargs={"dtype": torch.float16},
            )
        return self._model
