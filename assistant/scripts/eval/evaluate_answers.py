from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import unicodedata
from typing import Dict, List, Set

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from assistant.core.config import load_config
from assistant.rag.pipeline import RagPipeline


def _load_questions(path: Path) -> List[dict]:
	return json.loads(path.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
	text = text.strip().lower()
	text = "".join(
		ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
	)
	return " ".join(text.split())


def _tokenize(text: str) -> Set[str]:
	stopwords = {
		"de",
		"del",
		"la",
		"las",
		"el",
		"los",
		"y",
		"o",
		"en",
		"por",
		"para",
		"que",
		"se",
		"su",
		"sus",
		"un",
		"una",
		"uno",
		"unas",
		"unos",
		"a",
		"al",
	}
	words = _normalize(text).split()
	return {word for word in words if len(word) >= 4 and word not in stopwords}


def _match_fact(answer: str, fact: str) -> bool:
	answer_tokens = _tokenize(answer)
	fact_tokens = _tokenize(fact)
	if not fact_tokens:
		return False

	overlap = len(answer_tokens.intersection(fact_tokens))
	threshold = max(2, int(round(len(fact_tokens) * 0.5)))
	return overlap >= threshold


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


def _score_expected_facts(answer: str, expected_facts: List[str]) -> float:
	if not expected_facts:
		return 0.0
	matched = sum(1 for fact in expected_facts if _match_fact(answer, fact))
	return matched / len(expected_facts)


def _update_bucket(bucket: Dict[str, List[float]], key: str, value: float) -> None:
	bucket.setdefault(key, []).append(value)


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate RAG answers with expected_facts")
	parser.add_argument(
		"--questions",
		default="data/evaluation/datasets/unir_questions.json",
		help="Path to unir_questions.json",
	)
	parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
	parser.add_argument(
		"--use-reranking",
		action="store_true",
		help="Enable reranking of retrieved chunks",
	)
	parser.add_argument(
		"--use-orchestrator",
		action="store_true",
		help="Enable orchestrator flow for the pipeline",
	)
	parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
	args = parser.parse_args()

	questions_path = Path(args.questions)
	if not questions_path.exists():
		raise SystemExit(f"Questions file not found: {questions_path}")

	config = load_config()
	pipeline = RagPipeline(config)
	items = _load_questions(questions_path)
	if args.limit is not None:
		items = items[: args.limit]

	fact_scores: List[float] = []
	refusal_scores: List[int] = []
	by_type: Dict[str, List[float]] = {}
	by_category: Dict[str, List[float]] = {}
	failed = 0

	for item in items:
		question = item["question"].strip()
		answerable = bool(item.get("answerable", True))
		expected_facts = item.get("expected_facts", [])
		type_name = str(item.get("type", "unknown"))
		category = str(item.get("category", "unknown"))

		try:
			response = pipeline.run_pipeline(
				question,
				top_k=args.top_k,
				use_reranking=args.use_reranking,
				use_orchestrator=args.use_orchestrator,
			)
		except Exception as exc:
			failed += 1
			print(f"Skipped due to error: {question} -> {exc}")
			continue

		answer = response.answer
		if not answerable:
			refusal_scores.append(1 if _is_refusal(answer) else 0)
			_update_bucket(by_type, type_name, refusal_scores[-1])
			_update_bucket(by_category, category, refusal_scores[-1])
			continue

		if expected_facts:
			score = _score_expected_facts(answer, expected_facts)
			fact_scores.append(score)
			_update_bucket(by_type, type_name, score)
			_update_bucket(by_category, category, score)

	if fact_scores:
		avg_fact_score = sum(fact_scores) / len(fact_scores)
		print(f"Expected facts coverage (answerable only): {avg_fact_score:.3f}")
		print(f"Questions with expected_facts: {len(fact_scores)}")
	else:
		print("No expected_facts entries to score.")

	if refusal_scores:
		refusal_rate = sum(refusal_scores) / len(refusal_scores)
		print(f"Refusal rate (unanswerable): {refusal_rate:.3f}")
		print(f"Unanswerable questions: {len(refusal_scores)}")
	else:
		print("No unanswerable questions to score.")

	if by_type:
		print("\nBy type")
		for key, values in sorted(by_type.items()):
			avg = sum(values) / len(values)
			print(f"- {key}: {avg:.3f} (n={len(values)})")

	if by_category:
		print("\nBy category")
		for key, values in sorted(by_category.items()):
			avg = sum(values) / len(values)
			print(f"- {key}: {avg:.3f} (n={len(values)})")

	if failed:
		print(f"\nSkipped questions due to errors: {failed}")


if __name__ == "__main__":
	main()
