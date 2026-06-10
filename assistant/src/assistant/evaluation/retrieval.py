from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

from assistant.core.config import AppConfig
from assistant.rag.reranker import Reranker
from assistant.rag.retriever import Retriever


def load_questions(path: Path) -> List[dict]:
    """Load a retrieval dataset from JSON and return the item list."""
    return json.loads(path.read_text(encoding="utf-8"))


def _basename(value: str) -> str:
    """Return a normalized basename for a URL or file path."""
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        path = parsed.path.rstrip("/")
        if not path:
            return parsed.netloc
        return Path(path).name

    return Path(value).name


def _retrieve_results(
    question: str,
    retriever: Retriever,
    reranker: Reranker,
    config: AppConfig,
    top_k: int,
    use_hybrid: bool,
    use_reranking: bool,
) -> List:
    """Retrieve and optionally rerank candidate chunks for a query."""
    if use_reranking:
        candidates = retriever.retrieve_hybrid(
            question,
            top_k=top_k,
            candidate_k=config.rag_candidate_k,
        )
        return reranker.rerank(question, candidates, top_k=top_k)
    if use_hybrid:
        return retriever.retrieve_hybrid(
            question,
            top_k=top_k,
            candidate_k=config.rag_candidate_k,
        )
    return retriever.retrieve(question, top_k=top_k)


def _calculate_ranking_metrics(hit1: int, hit3: int, mrr_total: float, total: int) -> Dict[str, float]:
    """Aggregate retrieval ranking counters into normalized metrics."""
    return {
        "questions": total,
        "skipped": 0,
        "hit1": hit1 / total if total else 0.0,
        "hit3": hit3 / total if total else 0.0,
        "mrr": mrr_total / total if total else 0.0,
    }


def evaluate_retrieval(
    questions_path: Path,
    config: AppConfig,
    top_k: int = 5,
    use_hybrid: bool = False,
    use_reranking: bool = False,
) -> tuple[Dict[str, float], List[dict[str, object]]]:
    """Evaluate retrieval ranking metrics for a dataset of questions."""
    questions_path = questions_path.resolve()
    items = load_questions(questions_path)
    if not items:
        raise ValueError("No questions found in the dataset")

    retriever = Retriever(config)
    reranker = Reranker(config)

    hit1 = 0
    hit3 = 0
    mrr_total = 0.0
    skipped = 0
    rows: List[dict[str, object]] = []

    for item in items:
        question = str(item["question"]).strip()
        expected_single = str(item.get("expected_source", "")).strip()
        expected_sources = item.get("expected_sources") or ([] if not expected_single else [expected_single])
        answerable = bool(item.get("answerable", True))
        skipped_item = not answerable or not expected_sources

        results = []
        matched_rank = None
        if not skipped_item:
            results = _retrieve_results(
                question,
                retriever,
                reranker,
                config,
                top_k=top_k,
                use_hybrid=use_hybrid,
                use_reranking=use_reranking,
            )

            # Deduplicate retrieved chunks by document_id (keep first occurrence)
            seen_docs = set()
            dedup_results = []
            for chunk in results:
                doc_id = getattr(chunk, "document_id", None) or getattr(chunk, "url", chunk.source)
                if doc_id in seen_docs:
                    continue
                seen_docs.add(doc_id)
                dedup_results.append(chunk)

            for rank, chunk in enumerate(dedup_results, start=1):
                chunk_name = _basename(chunk.source)
                if any(chunk_name == expected for expected in expected_sources):
                    matched_rank = rank
                    break

            if matched_rank == 1:
                hit1 += 1
            if matched_rank is not None and matched_rank <= 3:
                hit3 += 1
            if matched_rank is not None:
                mrr_total += 1.0 / matched_rank
        else:
            skipped += 1

        rows.append(
            {
                "question": question,
                "expected_sources": expected_sources,
                "answerable": answerable,
                "use_hybrid": use_hybrid,
                "use_reranking": use_reranking,
                "top_k": top_k,
                "skipped": skipped_item,
                "matched_rank": matched_rank,
                "hit1": matched_rank == 1,
                "hit3": matched_rank is not None and matched_rank <= 3,
                "mrr": 1.0 / matched_rank if matched_rank else 0.0,
                "source_names": [chunk.source for chunk in (dedup_results if not skipped_item else results)],
                "source_scores": [chunk.score for chunk in (dedup_results if not skipped_item else results)],
            }
        )

    total = len(items) - skipped
    if total <= 0:
        raise ValueError("No valid questions available after filtering")

    metrics = _calculate_ranking_metrics(hit1, hit3, mrr_total, total)
    metrics["skipped"] = skipped
    # Save skipped questions for inspection
    if skipped > 0:
        skipped_rows = [r for r in rows if r.get("skipped")]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("assistant/results") / f"skipped_questions_{timestamp}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for r in skipped_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Skipped questions written to {out_path}")
    return metrics, rows
