from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from assistant.evaluation.utils import is_refusal, load_json
from assistant.rag.pipeline import RagPipeline


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Persist a list of dictionaries as a newline-delimited JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_orchestrator_result_row(
    item: dict,
    response: Any,
    is_orchestrator: bool,
) -> Dict[str, Any]:
    """Build a single result row from a pipeline response and dataset item."""
    question = item["question"].strip()
    expected_type = str(item.get("type", "unknown"))
    plan = response.plan
    predicted_type = plan.question_type if plan else None
    routing_correct = predicted_type == expected_type if plan else None
    refused = response.refused or is_refusal(response.answer)
    tool_calls_count = len(response.tool_calls or [])
    subquery_count = len(plan.subqueries) if plan else 0
    source_names = [chunk.source for chunk in (response.sources or [])]
    source_scores = [chunk.score for chunk in (response.sources or [])]

    return {
        "dataset": "main",
        "mode": "orchestrator" if is_orchestrator else "deterministic",
        "question": question,
        "expected_type": expected_type,
        "predicted_type": predicted_type,
        "routing_correct": routing_correct,
        "answerable": bool(item.get("answerable", True)),
        "refused": refused,
        "tool_calls": tool_calls_count,
        "subqueries": subquery_count,
        "intent": plan.intent if plan else None,
        "answer": response.answer,
        "source_names": source_names,
        "source_scores": source_scores,
        "retrieved_count": len(response.sources or []),
        "plan_subqueries": plan.subqueries if plan else [],
        "tool_call_details": [
            {
                "tool_name": call.tool_name,
                "tool_input": call.tool_input,
                "tool_output_summary": call.tool_output_summary,
            }
            for call in (response.tool_calls or [])
        ],
    }


def _evaluate_mode(
    items: List[dict],
    pipeline: RagPipeline,
    *,
    mode: str,
    top_k: int,
    use_hybrid: bool,
    use_reranking: bool,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate a set of orchestrator questions under the selected mode."""
    results: List[Dict[str, Any]] = []
    is_orchestrator = mode == "orchestrator"

    routing_hits = 0
    routing_total = 0
    refusal_hits = 0
    refusal_total = 0
    tool_calls_total = 0
    subquery_hits = 0
    subquery_total = 0

    for item in items:
        question = item["question"].strip()
        expected_type = str(item.get("type", "unknown"))
        answerable = bool(item.get("answerable", True))
        response = pipeline.run_pipeline(
            question,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_reranking=use_reranking,
            use_orchestrator=is_orchestrator,
        )

        plan = response.plan
        predicted_type = plan.question_type if plan else None
        routing_correct = predicted_type == expected_type if plan else None
        refused = response.refused or is_refusal(response.answer)

        if not answerable:
            refusal_total += 1
            refusal_hits += 1 if refused else 0

        tool_calls_count = len(response.tool_calls or [])
        tool_calls_total += tool_calls_count

        subquery_count = len(plan.subqueries) if plan else 0
        if expected_type in {"multi_doc", "comparative"}:
            subquery_total += 1
            subquery_hits += 1 if subquery_count > 1 else 0

        if plan:
            routing_total += 1
            routing_hits += 1 if routing_correct else 0

        results.append(
            _build_orchestrator_result_row(item, response, is_orchestrator)
        )

    metrics = {
        "mode": mode,
        "routing_accuracy": routing_hits / routing_total if routing_total else None,
        "refusal_rate": refusal_hits / refusal_total if refusal_total else None,
        "avg_tool_calls": tool_calls_total / len(items) if items else 0.0,
        "multi_subquery_rate": subquery_hits / subquery_total if subquery_total else None,
    }
    return metrics, results


def _evaluate_email_detection(
    email_items: List[dict],
    pipeline: RagPipeline,
    *,
    top_k: int,
    use_hybrid: bool,
    use_reranking: bool,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate whether the orchestrator correctly detects email-related actions."""
    results: List[Dict[str, Any]] = []
    correct = 0

    for item in email_items:
        question = item["question"].strip()
        expected = bool(item.get("expected_email_action", False))
        response = pipeline.run_pipeline(
            question,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_reranking=use_reranking,
            use_orchestrator=True,
        )
        predicted = any(
            tc.tool_name == "redactar_correo"
            for tc in (response.tool_calls or [])
        )
        correct += 1 if predicted == expected else 0

        results.append(
            {
                "dataset": "email",
                "mode": "orchestrator",
                "question": question,
                "expected_email_action": expected,
                "predicted_email_action": predicted,
            }
        )

    accuracy = correct / len(email_items) if email_items else None
    metrics = {
        "email_detection_accuracy": accuracy,
        "email_questions": len(email_items),
    }
    return metrics, results


def evaluate_orchestrator(
    questions_path: Path,
    email_questions_path: Path,
    pipeline: RagPipeline,
    top_k: int = 5,
    use_hybrid: bool = False,
    use_reranking: bool = False,
    compare: bool = False,
    mode: str | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Evaluate orchestrator behavior and email action detection over the dataset."""
    items = load_json(questions_path)
    email_items = load_json(email_questions_path)

    metrics_rows: List[Dict[str, Any]] = []
    result_rows: List[Dict[str, Any]] = []
    results_by_mode: Dict[str, List[Dict[str, Any]]] = {}

    if compare and mode:
        raise ValueError("Use either compare or mode, not both.")

    if compare:
        for selected_mode in ["deterministic", "orchestrator"]:
            metrics, rows = _evaluate_mode(
                items,
                pipeline,
                mode=selected_mode,
                top_k=top_k,
                use_hybrid=use_hybrid,
                use_reranking=use_reranking,
            )
            metrics_rows.append(metrics)
            result_rows.extend(rows)
            results_by_mode[selected_mode] = rows
    else:
        selected_mode = mode or "orchestrator"
        metrics, rows = _evaluate_mode(
            items,
            pipeline,
            mode=selected_mode,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_reranking=use_reranking,
        )
        metrics_rows.append(metrics)
        result_rows.extend(rows)
        results_by_mode[selected_mode] = rows

    email_metrics, email_rows = _evaluate_email_detection(
        email_items,
        pipeline,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_reranking=use_reranking,
    )
    metrics_rows.append(email_metrics)
    result_rows.extend(email_rows)

    return metrics_rows, result_rows, results_by_mode
