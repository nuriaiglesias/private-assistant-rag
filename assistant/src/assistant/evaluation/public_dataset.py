from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from assistant.core.config import AppConfig

_RRF_K = 60

_SPANISH_STOPWORDS = frozenset({
    "a", "al", "ante", "bajo", "con", "contra", "de", "del", "desde",
    "durante", "e", "el", "en", "entre", "es", "esa", "ese", "eso",
    "esta", "este", "esto", "hacia", "hasta", "la", "las", "le", "les",
    "lo", "los", "mas", "mediante", "mi", "no", "o", "para", "pero",
    "por", "que", "se", "según", "si", "sin", "sobre", "su", "sus",
    "también", "te", "tu", "tus", "u", "un", "una", "unas", "unos",
    "y", "yo",
})


def _tokenize(text: str) -> List[str]:
    """Tokenize Spanish text into meaningful lexical units for BM25 retrieval."""
    return [
        token for token in re.split(r"\W+", text.lower())
        if token and token not in _SPANISH_STOPWORDS and len(token) > 1
    ]


def _cosine_sim(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """Compute cosine similarity scores between a query vector and document vectors."""
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    d_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return d_norms @ q_norm


def _retrieve_semantic(query_vec: np.ndarray, doc_vecs: np.ndarray, top_k: int) -> List[int]:
    """Return the top-k document indices by dense semantic similarity."""
    sims = _cosine_sim(query_vec, doc_vecs)
    return list(np.argsort(sims)[::-1][:top_k])


def _retrieve_bm25(query: str, bm25: BM25Okapi, top_k: int) -> List[int]:
    """Return the top-k document indices ranked by BM25 score."""
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    return list(np.argsort(scores)[::-1][:top_k])


def _retrieve_hybrid_rrf(
    query: str,
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    bm25: BM25Okapi,
    candidate_k: int,
    top_k: int,
) -> List[int]:
    """Combine dense and lexical rankings using Reciprocal Rank Fusion."""
    dense_ranked = _retrieve_semantic(query_vec, doc_vecs, candidate_k)
    bm25_ranked = _retrieve_bm25(query, bm25, candidate_k)

    dense_rank = {idx: rank + 1 for rank, idx in enumerate(dense_ranked)}
    bm25_rank = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked)}

    fused: dict[int, float] = {}
    for idx in set(dense_rank) | set(bm25_rank):
        rrf_d = 0.5 / (_RRF_K + dense_rank[idx]) if idx in dense_rank else 0.0
        rrf_b = 0.5 / (_RRF_K + bm25_rank[idx]) if idx in bm25_rank else 0.0
        fused[idx] = rrf_d + rrf_b

    ranked = sorted(fused, key=lambda i: fused[i], reverse=True)
    return ranked[:top_k]


def _compute_metrics(results: List[Tuple[int, List[int]]]) -> dict:
    """Compute hit@1, hit@3 and MRR for a collection of retrieval results."""
    hit1 = hit3 = mrr_sum = 0.0
    n = len(results)

    for gold_idx, retrieved in results:
        rank = None
        for pos, idx in enumerate(retrieved, start=1):
            if idx == gold_idx:
                rank = pos
                break
        if rank == 1:
            hit1 += 1
        if rank is not None and rank <= 3:
            hit3 += 1
        if rank is not None:
            mrr_sum += 1.0 / rank

    return {
        "hit1": hit1 / n,
        "hit3": hit3 / n,
        "mrr": mrr_sum / n,
        "n": n,
    }


def _load_public_dataset(limit: int | None):
    """Load the Spanish XQuAD evaluation dataset with an optional limit."""
    dataset = load_dataset("xquad", "xquad.es", split="validation", trust_remote_code=True)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def _build_context_index(dataset) -> tuple[List[str], dict[str, int]]:
    """Build the unique context list and mapping from context text to index."""
    context_list = list(dict.fromkeys(dataset["context"]))
    return context_list, {ctx: i for i, ctx in enumerate(context_list)}


def evaluate_public_dataset(
    config: AppConfig,
    mode: str = "semantic",
    limit: int | None = None,
    top_k: int = 5,
    candidate_k: int = 20,
) -> dict:
    """Evaluate retrieval quality on the Spanish XQuAD dataset."""
    model = SentenceTransformer(config.embedding_model)
    dataset = _load_public_dataset(limit)
    context_list, context_index = _build_context_index(dataset)
    n_contexts = len(context_list)

    doc_vecs = np.array(
        model.encode(context_list, show_progress_bar=True, batch_size=32),
        dtype=np.float32,
    )

    bm25: BM25Okapi | None = None
    if mode == "hybrid":
        tokenized = [_tokenize(ctx) for ctx in context_list]
        bm25 = BM25Okapi(tokenized)

    query_texts = [ex["question"] for ex in dataset]
    query_vecs = model.encode(query_texts, show_progress_bar=True, batch_size=32)

    results: List[Tuple[int, List[int]]] = []
    for i, example in enumerate(dataset):
        gold_context = example["context"]
        gold_idx = context_index.get(gold_context)
        if gold_idx is None:
            continue

        query_vec = query_vecs[i]
        if mode == "hybrid" and bm25 is not None:
            retrieved = _retrieve_hybrid_rrf(
                query=example["question"],
                query_vec=query_vec,
                doc_vecs=doc_vecs,
                bm25=bm25,
                candidate_k=candidate_k,
                top_k=top_k,
            )
        else:
            retrieved = _retrieve_semantic(query_vec, doc_vecs, top_k=top_k)

        results.append((gold_idx, retrieved))

    metrics = _compute_metrics(results)
    metrics["mode"] = mode
    metrics["top_k"] = top_k
    metrics["candidate_k"] = candidate_k
    metrics["questions"] = len(dataset)
    metrics["contexts"] = n_contexts
    return metrics
