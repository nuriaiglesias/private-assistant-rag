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
        help="Enable hybrid retrieval (dense + BM25) without reranking — variant B",
    )
    parser.add_argument(
        "--use-reranking",
        action="store_true",
        help="Enable hybrid retrieval + reranking (implies hybrid) — variant C",
    )
    parser.add_argument(
        "--use-orchestrator",
        action="store_true",
        help="Enable orchestrator flow — combine with --use-reranking for variant D",
    )
    parser.add_argument(
        "--use-query-rewriting",
        action="store_true",
        help="Rewrite each question with the LLM before retrieval to improve recall",
    )
    parser.add_argument(
        "--output",
        default="assistant/results/answers_results.jsonl",
        help="Path to JSONL output for per-question answer evaluation details",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    parser.add_argument(
        "--skip-bertscore",
        action="store_true",
        help="Skip BERTScore computation to reduce memory usage (saves ~440MB)",
    )
    args = parser.parse_args()

    default_output = Path("assistant/results/answers_results.jsonl")
    output_path = Path(args.output)
    if args.output == str(default_output):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = default_output.with_name(
            f"{default_output.stem}_{timestamp}{default_output.suffix}"
        )

    questions_path = Path(args.questions)
    if not questions_path.exists():
        raise SystemExit(f"Questions file not found: {questions_path}")

    config = load_config()
    pipeline = RagPipeline(config)

    metrics, rows = evaluate_answers(
        questions_path=questions_path,
        pipeline=pipeline,
        top_k=args.top_k,
        use_hybrid=args.use_hybrid,
        use_reranking=args.use_reranking,
        use_orchestrator=args.use_orchestrator,
        use_query_rewriting=args.use_query_rewriting,
        limit=args.limit,
        skip_bertscore=args.skip_bertscore,
    )

    _write_jsonl(output_path, rows)
    print(f"Results written to {output_path}")

    if not metrics["rouge_available"]:
        print("Warning: rouge-score not installed — ROUGE-L not computed (pip install rouge-score)")
    if not metrics["bertscore_available"]:
        print("Warning: bert-score not installed — BERTScore not computed (pip install bert-score)")

    print(f"\nQuestions evaluated: {metrics['expected_facts_count'] + metrics['unanswerable_count']}")

    if metrics["avg_fact_score"] is not None:
        print(f"\n--- Answer quality (answerable questions: {metrics['expected_facts_count']}) ---")
        print(f"Token-overlap fact coverage : {metrics['avg_fact_score']:.3f}")
        if metrics["avg_rouge_l"] is not None:
            print(f"ROUGE-L F1                 : {metrics['avg_rouge_l']:.3f}")
        if metrics["avg_bertscore_f1"] is not None:
            print(f"BERTScore F1               : {metrics['avg_bertscore_f1']:.3f}")
    else:
        print("No expected_facts entries to score.")

    if metrics["refusal_rate"] is not None:
        print(f"\n--- Abstention (unanswerable questions: {metrics['unanswerable_count']}) ---")
        print(f"Refusal rate: {metrics['refusal_rate']:.3f}")

    if metrics["by_type"]:
        print("\n--- Token-overlap by question type ---")
        for key, value in sorted(metrics["by_type"].items()):
            print(f"  {key}: {value:.3f}")

    if metrics["by_category"]:
        print("\n--- Token-overlap by category ---")
        for key, value in sorted(metrics["by_category"].items()):
            print(f"  {key}: {value:.3f}")

    if metrics["failed"]:
        print(f"\nSkipped due to errors: {metrics['failed']}")


if __name__ == "__main__":
    main()
