from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from assistant.core.config import load_config
from assistant.evaluation.answers import evaluate_answers
from assistant.rag.pipeline import RagPipeline


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG answers with expected_facts")
    parser.add_argument(
        "--questions",
        default="assistant/datasets/unir_questions.json",
        help="Path to unir_questions.json",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    parser.add_argument(
        "--use-hybrid",
        action="store_true",
        help="Enable hybrid retrieval (dense + BM25) without reranking",
    )
    parser.add_argument(
        "--use-reranking",
        action="store_true",
        help="Enable reranking of retrieved chunks (implies hybrid retrieval)",
    )
    parser.add_argument(
        "--use-orchestrator",
        action="store_true",
        help="Enable orchestrator flow for the pipeline",
    )
    parser.add_argument(
        "--output",
        default="assistant/results/answers_results.jsonl",
        help="Path to JSONL output for per-question answer evaluation details",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    args = parser.parse_args()

    default_output = Path("assistant/results/answers_results.jsonl")
    output_path = Path(args.output)
    if args.output == str(default_output):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = default_output.with_name(f"{default_output.stem}_{timestamp}{default_output.suffix}")

    questions_path = Path(args.questions)
    if not questions_path.exists():
        raise SystemExit(f"Questions file not found: {questions_path}")

    config = load_config()
    pipeline = RagPipeline(config)

    metrics, rows = evaluate_answers(
        questions_path=questions_path,
        pipeline=pipeline,
        top_k=args.top_k,
        use_reranking=args.use_reranking,
        use_orchestrator=args.use_orchestrator,
        limit=args.limit,
    )

    _write_jsonl(output_path, rows)
    print(f"Results written to {output_path}")

    if metrics["avg_fact_score"] is not None:
        print(f"Expected facts coverage (answerable only): {metrics['avg_fact_score']:.3f}")
        print(f"Questions with expected_facts: {metrics['expected_facts_count']}")
    else:
        print("No expected_facts entries to score.")

    if metrics["refusal_rate"] is not None:
        print(f"Refusal rate (unanswerable): {metrics['refusal_rate']:.3f}")
        print(f"Unanswerable questions: {metrics['unanswerable_count']}")
    else:
        print("No unanswerable questions to score.")

    if metrics["by_type"]:
        print("\nBy type")
        for key, value in sorted(metrics["by_type"].items()):
            print(f"- {key}: {value:.3f} (n={len(metrics['by_type'][key])})")

    if metrics["by_category"]:
        print("\nBy category")
        for key, value in sorted(metrics["by_category"].items()):
            print(f"- {key}: {value:.3f} (n={len(metrics['by_category'][key])})")

    if metrics["failed"]:
        print(f"\nSkipped questions due to errors: {metrics['failed']}")


if __name__ == "__main__":
    main()
