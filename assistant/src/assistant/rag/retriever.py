from __future__ import annotations

"""Retriever implementation for semantic, lexical, and hybrid search."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from assistant.core.config import AppConfig
from assistant.core.paths import REPO_ROOT
from assistant.ingestion.document_loader import load_documents
from assistant.rag.chunker import chunk_text
from assistant.rag.vector_store import QdrantVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    score: float
    title: str = "unknown"
    url: str = "unknown"
    document_id: str = "unknown"
    chunk_id: str = "unknown"
    rerank_score: float | None = None


class Retriever:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model: SentenceTransformer | None = None
        self._store: QdrantVectorStore | None = None
        self._bm25_index: _BM25Index | None = None

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        query_vector = self._get_model().encode([query])[0].tolist()
        store = self._get_store(vector_size=len(query_vector))
        results = store.search(query_vector, limit=top_k)

        chunks: List[RetrievedChunk] = []
        for result in results:
            payload = result.payload or {}
            chunks.append(_chunk_from_payload(payload, score=float(result.score)))

        return chunks

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
    ) -> List[RetrievedChunk]:
        candidate_size = candidate_k or top_k

        dense_candidates = self.retrieve(query, top_k=candidate_size)
        bm25_candidates = self._get_bm25_index().search(
            query,
            top_k=candidate_size,
        )

        combined = _merge_candidates(
            dense_candidates,
            bm25_candidates,
            dense_weight=self._config.hybrid_dense_weight,
            bm25_weight=self._config.hybrid_bm25_weight,
        )

        return combined[:top_k]

    def retrieve_lexical(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        return self._get_bm25_index().search(query, top_k=top_k)

    def _get_store(self, vector_size: int) -> QdrantVectorStore:
        if self._store is None:
            self._store = QdrantVectorStore(
                url=self._config.qdrant_url,
                collection=self._config.qdrant_collection,
                vector_size=vector_size,
            )

        return self._store

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._config.embedding_model)

        return self._model

    def _get_bm25_index(self) -> "_BM25Index":
        if self._bm25_index is None:
            corpus_path = Path(self._config.hybrid_corpus_path)

            if not corpus_path.is_absolute():
                corpus_path = (REPO_ROOT / corpus_path).resolve()

            self._bm25_index = _BM25Index.from_corpus(
                corpus_path,
                self._config,
            )

        return self._bm25_index


@dataclass(frozen=True)
class _Chunk:
    text: str
    source: str
    chunk_index: int
    title: str = "unknown"
    url: str = "unknown"
    document_id: str = "unknown"
    chunk_id: str = "unknown"


class _BM25Index:
    def __init__(self, chunks: List[_Chunk], bm25: BM25Okapi) -> None:
        self._chunks = chunks
        self._bm25 = bm25

    @classmethod
    def from_corpus(cls, corpus_path: Path, config: AppConfig) -> "_BM25Index":
        documents = load_documents(corpus_path)

        chunks: List[_Chunk] = []
        tokenized_chunks: List[List[str]] = []

        for document_index, document in enumerate(documents):
            text_chunks = chunk_text(
                document.text,
                chunk_size=config.chunk_size,
                overlap=config.chunk_overlap,
            )

            source = _pick_source(document.metadata)
            title = str(document.metadata.get("title", "unknown"))
            url = str(
                document.metadata.get("url")
                or document.metadata.get("source_url")
                or source
            )
            document_id = str(
                document.metadata.get("document_id")
                or document.metadata.get("id")
                or f"doc_{document_index}"
            )

            for chunk_index, chunk in enumerate(text_chunks):
                chunk_id = str(
                    document.metadata.get("chunk_id")
                    or f"{document_id}_{chunk_index}"
                )

                chunks.append(
                    _Chunk(
                        text=chunk,
                        source=source,
                        chunk_index=chunk_index,
                        title=title,
                        url=url,
                        document_id=document_id,
                        chunk_id=chunk_id,
                    )
                )
                tokenized_chunks.append(_tokenize(chunk))

        bm25 = BM25Okapi(tokenized_chunks)
        return cls(chunks, bm25)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )

        results: List[RetrievedChunk] = []

        for index, score in ranked[:top_k]:
            chunk = self._chunks[index]
            results.append(
                RetrievedChunk(
                    text=chunk.text,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    score=float(score),
                    title=chunk.title,
                    url=chunk.url,
                    document_id=chunk.document_id,
                    chunk_id=chunk.chunk_id,
                )
            )

        return results


_SPANISH_STOPWORDS = frozenset(
    {
        "a",
        "al",
        "ante",
        "bajo",
        "con",
        "contra",
        "de",
        "del",
        "desde",
        "durante",
        "e",
        "el",
        "en",
        "entre",
        "es",
        "esa",
        "ese",
        "eso",
        "esta",
        "este",
        "esto",
        "hacia",
        "hasta",
        "la",
        "las",
        "le",
        "les",
        "lo",
        "los",
        "mas",
        "mediante",
        "mi",
        "no",
        "o",
        "para",
        "pero",
        "por",
        "que",
        "se",
        "según",
        "si",
        "sin",
        "sobre",
        "su",
        "sus",
        "también",
        "te",
        "tu",
        "tus",
        "u",
        "un",
        "una",
        "unas",
        "unos",
        "y",
        "yo",
    }
)


def _chunk_from_payload(payload: dict, score: float) -> RetrievedChunk:
    source = str(payload.get("source", "unknown"))

    return RetrievedChunk(
        text=str(payload.get("text", "")),
        source=source,
        chunk_index=_safe_int(payload.get("chunk_index"), default=0),
        score=score,
        title=str(payload.get("title", "unknown")),
        url=str(payload.get("url") or payload.get("source_url") or source),
        document_id=str(payload.get("document_id", "unknown")),
        chunk_id=str(payload.get("chunk_id", "unknown")),
    )


def _tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.split(r"\W+", text.lower())
        if token and token not in _SPANISH_STOPWORDS and len(token) > 1
    ]


def _pick_source(metadata: Dict) -> str:
    return str(
        metadata.get("source_url")
        or metadata.get("source_path")
        or metadata.get("source")
        or "unknown"
    )


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_RRF_K = 60


def _merge_candidates(
    dense: List[RetrievedChunk],
    bm25: List[RetrievedChunk],
    *,
    dense_weight: float,
    bm25_weight: float,
) -> List[RetrievedChunk]:
    def key(chunk: RetrievedChunk) -> Tuple[str, int, str]:
        return (
            chunk.source,
            chunk.chunk_index,
            chunk.text,
        )

    dense_map = {key(chunk): chunk for chunk in dense}
    bm25_map = {key(chunk): chunk for chunk in bm25}

    dense_rank = {
        item_key: rank + 1
        for rank, item_key in enumerate(dense_map)
    }
    bm25_rank = {
        item_key: rank + 1
        for rank, item_key in enumerate(bm25_map)
    }

    fused_scores: Dict[Tuple[str, int, str], float] = {}

    for item_key in set(dense_map) | set(bm25_map):
        rrf_dense = (
            dense_weight / (_RRF_K + dense_rank[item_key])
            if item_key in dense_rank
            else 0.0
        )
        rrf_bm25 = (
            bm25_weight / (_RRF_K + bm25_rank[item_key])
            if item_key in bm25_rank
            else 0.0
        )
        fused_scores[item_key] = rrf_dense + rrf_bm25

    merged: List[RetrievedChunk] = []

    for item_key, score in fused_scores.items():
        chunk = dense_map.get(item_key) or bm25_map[item_key]

        merged.append(
            RetrievedChunk(
                text=chunk.text,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=score,
                title=chunk.title,
                url=chunk.url,
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                rerank_score=chunk.rerank_score,
            )
        )

    return sorted(
        merged,
        key=lambda item: item.score,
        reverse=True,
    )