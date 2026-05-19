from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List
from urllib.parse import urlparse

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from assistant.core.config import load_config
from assistant.rag.reranker import Reranker
from assistant.rag.retriever import Retriever


def _load_questions(path: Path) -> List[dict]:
	return json.loads(path.read_text(encoding="utf-8"))


def _basename(value: str) -> str:
	parsed = urlparse(value)
	if parsed.scheme and parsed.netloc:
		path = parsed.path.rstrip("/")
		if not path:
			return parsed.netloc
		return Path(path).name

	return Path(value).name


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate retrieval with Hit@K and MRR")
	parser.add_argument(
		"--questions",
		default="data/evaluation/questions.json",
		help="Path to questions.json",
	)
	parser.add_argument("--top-k", type=int, default=5, help="Max results to retrieve")
	parser.add_argument(
		"--use-reranking",
		action="store_true",
		help="Use hybrid retrieval + reranking",
	)
	args = parser.parse_args()

	questions_path = Path(args.questions)
	if not questions_path.exists():
		raise SystemExit(f"Questions file not found: {questions_path}")

	config = load_config()
	retriever = Retriever(config)
	reranker = Reranker(config)
	items = _load_questions(questions_path)

	if not items:
		raise SystemExit("No questions found in the dataset")

	hit1 = 0
	hit3 = 0
	mrr_total = 0.0

	skipped = 0
	for item in items:
		question = item["question"].strip()
		expected_single = str(item.get("expected_source", "")).strip()
		expected_sources = item.get("expected_sources") or ([] if not expected_single else [expected_single])
		answerable = bool(item.get("answerable", True))
		if not answerable or not expected_sources:
			skipped += 1
			continue
		if args.use_reranking:
			candidates = retriever.retrieve_hybrid(
				question,
				top_k=args.top_k,
				candidate_k=config.rag_candidate_k,
			)
			results = reranker.rerank(question, candidates, top_k=args.top_k)
		else:
			results = retriever.retrieve(question, top_k=args.top_k)
		matched_rank = None

		for rank, chunk in enumerate(results, start=1):
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

	total = len(items) - skipped
	hit1_score = hit1 / total
	hit3_score = hit3 / total
	mrr_score = mrr_total / total

	print("Retrieval evaluation")
	print(f"Questions: {total} (skipped {skipped})")
	print(f"Hit@1: {hit1_score:.3f} ({hit1}/{total})")
	print(f"Hit@3: {hit3_score:.3f} ({hit3}/{total})")
	print(f"MRR:   {mrr_score:.3f}")


if __name__ == "__main__":
	main()
