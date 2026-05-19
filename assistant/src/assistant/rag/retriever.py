from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from sentence_transformers import SentenceTransformer

from assistant.core.config import AppConfig
from assistant.ingestion.document_loader import load_documents
from assistant.rag.chunker import chunk_text
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
		self._bm25_index: _BM25Index | None = None

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

	def retrieve_hybrid(self, query: str, top_k: int = 5, candidate_k: int | None = None) -> List[RetrievedChunk]:
		candidate_size = candidate_k or top_k
		dense_candidates = self.retrieve(query, top_k=candidate_size)
		bm25_candidates = self._get_bm25_index().search(query, top_k=candidate_size)

		combined = _merge_candidates(
			dense_candidates,
			bm25_candidates,
			dense_weight=self._config.hybrid_dense_weight,
			bm25_weight=self._config.hybrid_bm25_weight,
		)
		return combined[:candidate_size]

	def _get_store(self, vector_size: int) -> QdrantVectorStore:
		if self._store is None:
			self._store = QdrantVectorStore(
				url=self._config.qdrant_url,
				collection=self._config.qdrant_collection,
				vector_size=vector_size,
			)
		return self._store

	def _get_bm25_index(self) -> "_BM25Index":
		if self._bm25_index is None:
			corpus_path = Path(self._config.hybrid_corpus_path)
			if not corpus_path.is_absolute():
				repo_root = Path(__file__).resolve().parents[3]
				corpus_path = (repo_root / corpus_path).resolve()
			self._bm25_index = _BM25Index.from_corpus(corpus_path, self._config)
		return self._bm25_index


@dataclass(frozen=True)
class _Chunk:
	text: str
	source: str
	chunk_index: int


class _BM25Index:
	def __init__(self, chunks: List[_Chunk], bm25: BM25Okapi) -> None:
		self._chunks = chunks
		self._bm25 = bm25

	@classmethod
	def from_corpus(cls, corpus_path: Path, config: AppConfig) -> "_BM25Index":
		documents = load_documents(corpus_path)
		chunks: List[_Chunk] = []
		tokenized: List[List[str]] = []

		for doc in documents:
			text_chunks = chunk_text(
				doc.text,
				chunk_size=config.chunk_size,
				overlap=config.chunk_overlap,
			)
			source = _pick_source(doc.metadata)
			for index, chunk in enumerate(text_chunks):
				chunks.append(_Chunk(text=chunk, source=source, chunk_index=index))
				tokenized.append(_tokenize(chunk))

		bm25 = BM25Okapi(tokenized)
		return cls(chunks, bm25)

	def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
		query_tokens = _tokenize(query)
		scores = self._bm25.get_scores(query_tokens)
		ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
		results: List[RetrievedChunk] = []
		for index, score in ranked[:top_k]:
			chunk = self._chunks[index]
			results.append(
				RetrievedChunk(
					text=chunk.text,
					source=chunk.source,
					chunk_index=chunk.chunk_index,
					score=float(score),
				)
			)
		return results


def _tokenize(text: str) -> List[str]:
	return [token for token in re.split(r"\W+", text.lower()) if token]


def _pick_source(metadata: Dict) -> str:
	return (
		metadata.get("source_url")
		or metadata.get("source_path")
		or metadata.get("source")
		or "unknown"
	)


def _merge_candidates(
	dense: List[RetrievedChunk],
	bm25: List[RetrievedChunk],
	*,
	dense_weight: float,
	bm25_weight: float,
) -> List[RetrievedChunk]:
	def key(chunk: RetrievedChunk) -> Tuple[str, int, str]:
		return (chunk.source, chunk.chunk_index, chunk.text)

	def normalize(scores: Dict[Tuple[str, int, str], float]) -> Dict[Tuple[str, int, str], float]:
		if not scores:
			return {}
		values = list(scores.values())
		min_score = min(values)
		max_score = max(values)
		if max_score == min_score:
			return {item_key: 1.0 for item_key in scores}
		return {
			item_key: (value - min_score) / (max_score - min_score)
			for item_key, value in scores.items()
		}

	dense_map = {key(chunk): chunk for chunk in dense}
	bm25_map = {key(chunk): chunk for chunk in bm25}

	dense_scores = {item_key: chunk.score for item_key, chunk in dense_map.items()}
	bm25_scores = {item_key: chunk.score for item_key, chunk in bm25_map.items()}

	dense_norm = normalize(dense_scores)
	bm25_norm = normalize(bm25_scores)

	merged: List[RetrievedChunk] = []
	for item_key in set(dense_map) | set(bm25_map):
		chunk = dense_map.get(item_key) or bm25_map[item_key]
		score = dense_weight * dense_norm.get(item_key, 0.0) + bm25_weight * bm25_norm.get(item_key, 0.0)
		merged.append(
			RetrievedChunk(
				text=chunk.text,
				source=chunk.source,
				chunk_index=chunk.chunk_index,
				score=score,
			)
		)

	return sorted(merged, key=lambda item: item.score, reverse=True)
