from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unicodedata
from typing import Any, Dict, List

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from assistant.core.config import load_config
from assistant.rag.pipeline import RagPipeline


def _load_items(path: Path) -> List[dict]:
	return json.loads(path.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
	text = text.strip().lower()
	text = "".join(
		ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
	)
	return " ".join(text.split())


def _is_refusal(answer: str) -> bool:
	answer_norm = _normalize(answer)
	patterns = [
		"no dispongo de informacion",
		"no tengo informacion",
		"no hay informacion suficiente",
		"informacion insuficiente",
		"no puedo responder",
		"no se dispone de informacion",
	]
	return any(pattern in answer_norm for pattern in patterns)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		for row in rows:
			handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _evaluate_mode(
	items: List[dict],
	pipeline: RagPipeline,
	*,
	mode: str,
	top_k: int,
	use_reranking: bool,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
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
			use_reranking=use_reranking,
			use_orchestrator=is_orchestrator,
		)

		plan = response.plan
		predicted_type = plan.question_type if plan else None
		routing_correct = predicted_type == expected_type if plan else None

		refused = response.refused or _is_refusal(response.answer)
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
			{
				"dataset": "main",
				"mode": mode,
				"question": question,
				"expected_type": expected_type,
				"predicted_type": predicted_type,
				"routing_correct": routing_correct,
				"answerable": answerable,
				"refused": refused,
				"tool_calls": tool_calls_count,
				"subqueries": subquery_count,
				"intent": plan.intent if plan else None,
			}
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
	use_reranking: bool,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
	results: List[Dict[str, Any]] = []
	correct = 0

	for item in email_items:
		question = item["question"].strip()
		expected = bool(item.get("expected_email_action", False))
		response = pipeline.run_pipeline(
			question,
			top_k=top_k,
			use_reranking=use_reranking,
			use_orchestrator=True,
		)
		plan = response.plan
		predicted = plan.intent == "email_action" if plan else False
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


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate orchestrator routing and tool usage")
	parser.add_argument(
		"--questions",
		default="data/evaluation/unir_questions.json",
		help="Path to unir_questions.json",
	)
	parser.add_argument(
		"--email-questions",
		default="data/evaluation/email_questions.json",
		help="Path to email_questions.json",
	)
	parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
	parser.add_argument(
		"--use-reranking",
		action="store_true",
		help="Enable reranking of retrieved chunks",
	)
	parser.add_argument(
		"--compare",
		action="store_true",
		help="Compare deterministic vs orchestrator",
	)
	parser.add_argument(
		"--mode",
		choices=["deterministic", "orchestrator"],
		default=None,
		help="Run a single mode without comparison",
	)
	parser.add_argument(
		"--output",
		default="data/evaluation/orchestrator_results.jsonl",
		help="Path to JSONL output",
	)
	parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
	args = parser.parse_args()

	questions_path = Path(args.questions)
	if not questions_path.exists():
		raise SystemExit(f"Questions file not found: {questions_path}")

	email_path = Path(args.email_questions)
	if not email_path.exists():
		raise SystemExit(f"Email questions file not found: {email_path}")

	config = load_config()
	pipeline = RagPipeline(config)
	items = _load_items(questions_path)
	if args.limit is not None:
		items = items[: args.limit]

	metrics_rows: List[Dict[str, Any]] = []
	result_rows: List[Dict[str, Any]] = []
	results_by_mode: Dict[str, List[Dict[str, Any]]] = {}

	if args.compare and args.mode:
		raise SystemExit("Use either --compare or --mode, not both.")

	if args.compare:
		for mode in ["deterministic", "orchestrator"]:
			metrics, rows = _evaluate_mode(
				items,
				pipeline,
				mode=mode,
				top_k=args.top_k,
				use_reranking=args.use_reranking,
			)
			metrics_rows.append(metrics)
			result_rows.extend(rows)
			results_by_mode[mode] = rows
	else:
		selected_mode = args.mode or "orchestrator"
		metrics, rows = _evaluate_mode(
			items,
			pipeline,
			mode=selected_mode,
			top_k=args.top_k,
			use_reranking=args.use_reranking,
		)
		metrics_rows.append(metrics)
		result_rows.extend(rows)
		results_by_mode[selected_mode] = rows

	email_items = _load_items(email_path)
	email_metrics, email_rows = _evaluate_email_detection(
		email_items,
		pipeline,
		top_k=args.top_k,
		use_reranking=args.use_reranking,
	)
	metrics_rows.append(email_metrics)
	result_rows.extend(email_rows)

	output_path = Path(args.output)
	if args.compare:
		base = output_path.with_suffix("")
		for mode, rows in results_by_mode.items():
			mode_path = Path(f"{base}_{mode}.jsonl")
			_write_jsonl(mode_path, rows)
			print(f"Results written to {mode_path}")
		combined_path = Path(f"{base}_combined.jsonl")
		_write_jsonl(combined_path, result_rows)
		print(f"Results written to {combined_path}")
	else:
		if args.mode and output_path == Path("data/evaluation/orchestrator_results.jsonl"):
			output_path = Path(
				f"data/evaluation/orchestrator_results_{args.mode}.jsonl"
			)
		_write_jsonl(output_path, result_rows)
		print(f"Results written to {output_path}")

	print("Orchestrator evaluation")
	for row in metrics_rows:
		for key, value in row.items():
			print(f"{key}: {value}")
		print("-")


if __name__ == "__main__":
	main()
