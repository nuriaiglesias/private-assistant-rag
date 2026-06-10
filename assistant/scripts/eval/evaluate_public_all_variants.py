"""Run retrieval experiments A/B/C on XQuAD-es and prepare D (end-to-end) command.

This script computes semantic (A), hybrid RRF (B) and hybrid+reranker (C)
retrieval metrics on the XQuAD Spanish validation set. For variant D (orchestrator
and LLM) it generates a `questions.json` snippet (limitable) and prints the
command to run `evaluate_answers.py` since that step invokes the LLM and is
costly; running it is left to the user.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from assistant.core.config import load_config
from assistant.rag.reranker import Reranker
from assistant.rag.retriever import RetrievedChunk


_RRF_K = 60


def _tokenize(text: str) -> List[str]:
    return [t for t in __import__("re").split(r"\W+", text.lower()) if t and len(t) > 1]


def _cosine_sim(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    d_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
    return d_norms @ q_norm


def _retrieve_semantic(query_vec: np.ndarray, doc_vecs: np.ndarray, top_k: int) -> List[int]:
    sims = _cosine_sim(query_vec, doc_vecs)
    return list(np.argsort(sims)[::-1][:top_k])


def _retrieve_bm25(query: str, bm25: BM25Okapi, top_k: int) -> List[int]:
    tokens = _tokenize(query)
    scores = bm25.get_scores(tokens)
    return list(np.argsort(scores)[::-1][:top_k])


def _retrieve_hybrid_rrf(query: str, query_vec: np.ndarray, doc_vecs: np.ndarray, bm25: BM25Okapi, candidate_k: int, top_k: int) -> List[int]:
    dense_ranked = _retrieve_semantic(query_vec, doc_vecs, candidate_k)
    bm25_ranked = _retrieve_bm25(query, bm25, candidate_k)

    dense_rank = {idx: rank + 1 for rank, idx in enumerate(dense_ranked)}
    bm25_rank = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked)}

    fused = {}
    for idx in set(dense_rank) | set(bm25_rank):
        rrf_d = 0.5 / (_RRF_K + dense_rank[idx]) if idx in dense_rank else 0.0
        rrf_b = 0.5 / (_RRF_K + bm25_rank[idx]) if idx in bm25_rank else 0.0
        fused[idx] = rrf_d + rrf_b

    ranked = sorted(fused, key=lambda i: fused[i], reverse=True)
    return ranked[:top_k]


def _compute_metrics(results: List[Tuple[int, List[int]]]) -> dict:
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

    return {"hit1": hit1 / n, "hit3": hit3 / n, "mrr": mrr_sum / n, "n": n}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate XQuAD-es for variants A/B/C and prepare D")
    parser.add_argument("--limit", type=int, default=200, help="Limit number of questions for fast runs (default:200)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for evaluation")
    parser.add_argument("--candidate-k", type=int, default=20, help="Candidate pool size for hybrid/reranking")
    parser.add_argument("--mode", choices=["all", "A", "B", "C", "D"], default="all")
    args = parser.parse_args()

    config = load_config()
    model = SentenceTransformer(config.embedding_model)

    print("Loading XQuAD-es... (this may download the dataset)")
    dataset = load_dataset("xquad", "xquad.es", split="validation", trust_remote_code=True)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    context_list = list(dict.fromkeys(dataset["context"]))
    context_index = {ctx: i for i, ctx in enumerate(context_list)}

    print("Encoding contexts...")
    doc_vecs = np.array(model.encode(context_list, show_progress_bar=True, batch_size=32), dtype=np.float32)

    bm25 = None
    print("Building BM25 index...")
    tokenized = [_tokenize(ctx) for ctx in context_list]
    bm25 = BM25Okapi(tokenized)

    print("Encoding queries...")
    query_texts = [ex["question"] for ex in dataset]
    query_vecs = model.encode(query_texts, show_progress_bar=True, batch_size=32)

    results_A = []
    results_B = []
    results_C = []

    reranker = Reranker(config)

    for i, example in enumerate(dataset):
        gold_ctx = example["context"]
        gold_idx = context_index.get(gold_ctx)
        if gold_idx is None:
            continue

        q = example["question"]
        qvec = query_vecs[i]

        # A: semantic
        sem = _retrieve_semantic(qvec, doc_vecs, args.top_k)
        results_A.append((gold_idx, sem))

        # B: hybrid RRF
        hybrid = _retrieve_hybrid_rrf(q, qvec, doc_vecs, bm25, candidate_k=args.candidate_k, top_k=args.top_k)
        results_B.append((gold_idx, hybrid))

        # C: hybrid + reranker
        candidate_indices = _retrieve_hybrid_rrf(q, qvec, doc_vecs, bm25, candidate_k=args.candidate_k, top_k=args.candidate_k)
        candidates = []
        for idx in candidate_indices:
            candidates.append(
                RetrievedChunk(
                    text=context_list[idx],
                    source=f"ctx_{idx}",
                    chunk_index=0,
                    score=0.0,
                    title="xquad",
                    url=f"ctx://{idx}",
                    document_id=str(idx),
                    chunk_id=str(idx),
                )
            )

        if candidates:
            reranked = reranker.rerank(q, candidates, top_k=args.top_k)
            reranked_indices = [int(c.document_id) for c in reranked]
        else:
            reranked_indices = []

        results_C.append((gold_idx, reranked_indices))

    metrics_A = _compute_metrics(results_A)
    metrics_B = _compute_metrics(results_B)
    metrics_C = _compute_metrics(results_C)

    print("\n" + "=" * 40)
    print("XQuAD-es retrieval evaluation (subset)")
    print(f"Questions: {metrics_A['n']}")
    print("=" * 40)
    print(f"A (semantic)   - Hit@1: {metrics_A['hit1']:.3f}, Hit@3: {metrics_A['hit3']:.3f}, MRR: {metrics_A['mrr']:.3f}")
    print(f"B (hybrid RRF) - Hit@1: {metrics_B['hit1']:.3f}, Hit@3: {metrics_B['hit3']:.3f}, MRR: {metrics_B['mrr']:.3f}")
    print(f"C (hybrid+rer) - Hit@1: {metrics_C['hit1']:.3f}, Hit@3: {metrics_C['hit3']:.3f}, MRR: {metrics_C['mrr']:.3f}")
    print("=" * 40)

    # Prepare D: generate questions.json snippet for evaluate_answers.py on a small subset
    d_limit = min(50, len(dataset))
    questions = []
    for ex in dataset.select(range(d_limit)):
        expected_facts = []
        answers = ex.get("answers") or {}
        texts = answers.get("text") if isinstance(answers, dict) else None
        if texts:
            expected_facts = [texts[0]]

        questions.append({
            "question": ex["question"],
            "expected_facts": expected_facts,
            "answerable": True,
        })

    out_path = Path("assistant/results/xquad_questions_for_answers.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Prepared sample questions for variant D: {}".format(out_path))
    print("To run variant D (orchestrator + LLM) on this sample, execute:")
    print(
        "python3 assistant/scripts/eval/evaluate_answers.py \\\n"
        "  --questions assistant/results/xquad_questions_for_answers.json \\\n"
        "  --top-k 5 --use-hybrid --use-reranking --use-orchestrator --limit 50"
    )


if __name__ == "__main__":
    main()
