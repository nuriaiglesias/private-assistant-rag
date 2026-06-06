from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from assistant.evaluation.utils import is_refusal, load_json, tokenize_words
from assistant.rag.pipeline import RagPipeline


def _match_fact(answer: str, fact: str) -> bool:
    """Return True when the answer contains a sufficient token overlap with a fact."""
    answer_tokens = set(tokenize_words(answer))
    fact_tokens = set(tokenize_words(fact))
    if not fact_tokens:
        return False

    overlap = len(answer_tokens.intersection(fact_tokens))
    threshold = max(2, int(round(len(fact_tokens) * 0.5)))
    return overlap >= threshold


def _score_expected_facts(answer: str, expected_facts: List[str]) -> float:
    """Compute the fraction of expected facts present in the generated answer."""
    if not expected_facts:
        return 0.0
    matched = sum(1 for fact in expected_facts if _match_fact(answer, fact))
    return matched / len(expected_facts)


def _update_bucket(bucket: Dict[str, List[float]], key: str, value: float) -> None:
    """Add a numeric value to a bucket keyed by category or type."""
    bucket.setdefault(key, []).append(value)


def _evaluate_answer_item(
    item: dict,
    pipeline: RagPipeline,
    top_k: int,
    use_reranking: bool,
    use_orchestrator: bool,
) -> dict[str, object]:
    """Run the pipeline for a single evaluation item and return structured item metrics."""
    question = str(item["question"]).strip()
    answerable = bool(item.get("answerable", True))
    expected_facts = item.get("expected_facts", [])
    type_name = str(item.get("type", "unknown"))
    category = str(item.get("category", "unknown"))

    response = pipeline.run_pipeline(
        question,
        top_k=top_k,
        use_reranking=use_reranking,
        use_orchestrator=use_orchestrator,
    )

    result: dict[str, object] = {
        "question": question,
        "type": type_name,
        "category": category,
        "answerable": answerable,
        "use_reranking": use_reranking,
        "use_orchestrator": use_orchestrator,
        "top_k": top_k,
        "answer": response.answer,
        "source_names": [chunk.source for chunk in response.sources],
        "source_scores": [chunk.score for chunk in response.sources],
        "retrieved_count": len(response.sources),
        "plan": {
            "question_type": response.plan.question_type,
            "intent": response.plan.intent,
            "action": response.plan.action,
            "subqueries": response.plan.subqueries,
            "answer_style": response.plan.answer_style,
        }
        if response.plan
        else None,
        "tool_calls": [
            {
                "tool_name": tool.tool_name,
                "tool_input": tool.tool_input,
                "tool_output_summary": tool.tool_output_summary,
            }
            for tool in (response.tool_calls or [])
        ],
    }

    if not answerable:
        result["refusal"] = 1 if is_refusal(response.answer) else 0
        return result

    result["score"] = _score_expected_facts(response.answer, expected_facts) if expected_facts else None
    return result


def evaluate_answers(
    questions_path: Path,
    pipeline: RagPipeline,
    top_k: int = 5,
    use_reranking: bool = False,
    use_orchestrator: bool = False,
    limit: int | None = None,
) -> tuple[Dict[str, object], List[dict[str, object]]]:
    """Evaluate a question set by running the RAG pipeline and scoring expected facts."""
    items = load_json(questions_path)
    if limit is not None:
        items = items[:limit]

    fact_scores: List[float] = []
    refusal_scores: List[int] = []
    by_type: Dict[str, List[float]] = {}
    by_category: Dict[str, List[float]] = {}
    failed = 0
    rows: List[dict[str, object]] = []

    for item in items:
        try:
            evaluation = _evaluate_answer_item(
                item,
                pipeline,
                top_k=top_k,
                use_reranking=use_reranking,
                use_orchestrator=use_orchestrator,
            )
        except Exception:
            failed += 1
            continue

        rows.append(evaluation)

        if not evaluation["answerable"]:
            refusal_score = evaluation["refusal"]
            refusal_scores.append(refusal_score)
            _update_bucket(by_type, evaluation["type"], refusal_score)
            _update_bucket(by_category, evaluation["category"], refusal_score)
            continue

        score = evaluation.get("score")
        if score is not None:
            fact_scores.append(score)
            _update_bucket(by_type, evaluation["type"], score)
            _update_bucket(by_category, evaluation["category"], score)

    avg_fact_score = sum(fact_scores) / len(fact_scores) if fact_scores else None
    refusal_rate = sum(refusal_scores) / len(refusal_scores) if refusal_scores else None

    metrics = {
        "avg_fact_score": avg_fact_score,
        "expected_facts_count": len(fact_scores),
        "refusal_rate": refusal_rate,
        "unanswerable_count": len(refusal_scores),
        "by_type": {key: sum(values) / len(values) for key, values in by_type.items()},
        "by_category": {key: sum(values) / len(values) for key, values in by_category.items()},
        "failed": failed,
    }

    return metrics, rows
