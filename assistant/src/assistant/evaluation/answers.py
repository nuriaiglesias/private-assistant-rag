from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from assistant.evaluation.utils import is_refusal, load_json, tokenize_words
from assistant.rag.pipeline import RagPipeline

try:
    from rouge_score import rouge_scorer as _rouge_scorer_lib
    _rouge_scorer = _rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=False)
    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False

try:
    from bert_score import score as _bert_score_fn
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False


def _compute_rouge_l(answer: str, facts: List[str]) -> float | None:
    """Mean ROUGE-L F1 between the answer and each expected fact."""
    if not _ROUGE_AVAILABLE or not facts:
        return None
    scores = [
        _rouge_scorer.score(fact, answer)["rougeL"].fmeasure
        for fact in facts
    ]
    return sum(scores) / len(scores)


def _compute_bertscore(answer: str, facts: List[str], *, skip: bool = False) -> float | None:
    """Mean BERTScore F1 between the answer and each expected fact."""
    if skip or not _BERTSCORE_AVAILABLE or not facts:
        return None
    hypotheses = [answer] * len(facts)
    try:
        _, _, f1 = _bert_score_fn(hypotheses, facts, lang="es", verbose=False)
        return float(f1.mean())
    except Exception:
        return None


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
    """Fraction of expected facts present in the generated answer (token-overlap proxy)."""
    if not expected_facts:
        return 0.0
    matched = sum(1 for fact in expected_facts if _match_fact(answer, fact))
    return matched / len(expected_facts)


def _update_bucket(bucket: Dict[str, List[float]], key: str, value: float) -> None:
    bucket.setdefault(key, []).append(value)


def _evaluate_answer_item(
    item: dict,
    pipeline: RagPipeline,
    top_k: int,
    use_hybrid: bool,
    use_reranking: bool,
    use_orchestrator: bool,
    use_query_rewriting: bool,
    skip_bertscore: bool = False,
) -> dict[str, object]:
    """Run the pipeline for a single evaluation item and return structured metrics."""
    question = str(item["question"]).strip()
    answerable = bool(item.get("answerable", True))
    expected_facts = item.get("expected_facts", [])
    type_name = str(item.get("type", "unknown"))
    category = str(item.get("category", "unknown"))

    response = pipeline.run_pipeline(
        question,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_reranking=use_reranking,
        use_orchestrator=use_orchestrator,
        use_query_rewriting=use_query_rewriting,
    )

    result: dict[str, object] = {
        "question": question,
        "type": type_name,
        "category": category,
        "answerable": answerable,
        "use_hybrid": use_hybrid,
        "use_reranking": use_reranking,
        "use_orchestrator": use_orchestrator,
        "use_query_rewriting": use_query_rewriting,
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

    result["expected_facts"] = expected_facts

    if not answerable:
        result["refusal"] = 1 if is_refusal(response.answer) else 0
        return result

    if expected_facts:
        result["score"] = _score_expected_facts(response.answer, expected_facts)
        result["rouge_l"] = _compute_rouge_l(response.answer, expected_facts)
        result["bertscore_f1"] = _compute_bertscore(response.answer, expected_facts, skip=skip_bertscore)
    else:
        result["score"] = None
        result["rouge_l"] = None
        result["bertscore_f1"] = None

    return result


def evaluate_answers(
    questions_path: Path,
    pipeline: RagPipeline,
    top_k: int = 5,
    use_hybrid: bool = False,
    use_reranking: bool = False,
    use_orchestrator: bool = False,
    use_query_rewriting: bool = False,
    limit: int | None = None,
    skip_bertscore: bool = False,
) -> tuple[Dict[str, object], List[dict[str, object]]]:
    """Evaluate a question set and return aggregated metrics and per-question rows."""
    items = load_json(questions_path)
    if limit is not None:
        items = items[:limit]

    fact_scores: List[float] = []
    rouge_l_scores: List[float] = []
    bertscore_scores: List[float] = []
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
                use_hybrid=use_hybrid,
                use_reranking=use_reranking,
                use_orchestrator=use_orchestrator,
                use_query_rewriting=use_query_rewriting,
                skip_bertscore=skip_bertscore,
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

        rouge = evaluation.get("rouge_l")
        if rouge is not None:
            rouge_l_scores.append(rouge)

        bscore = evaluation.get("bertscore_f1")
        if bscore is not None:
            bertscore_scores.append(bscore)

    def _avg(values: List[float]) -> float | None:
        return sum(values) / len(values) if values else None

    metrics = {
        "avg_fact_score": _avg(fact_scores),
        "avg_rouge_l": _avg(rouge_l_scores),
        "avg_bertscore_f1": _avg(bertscore_scores),
        "expected_facts_count": len(fact_scores),
        "refusal_rate": _avg(refusal_scores),
        "unanswerable_count": len(refusal_scores),
        "by_type": {key: sum(values) / len(values) for key, values in by_type.items()},
        "by_category": {key: sum(values) / len(values) for key, values in by_category.items()},
        "failed": failed,
        "rouge_available": _ROUGE_AVAILABLE,
        "bertscore_available": _BERTSCORE_AVAILABLE,
    }

    return metrics, rows
