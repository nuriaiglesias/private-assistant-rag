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
from assistant.evaluation.retrieval import evaluate_retrieval


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval with Hit@K and MRR")
    parser.add_argument(
        "--questions",
        default="assistant/datasets/unir_questions.json",
        help="Path to unir_questions.json",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Max results to retrieve")
    parser.add_argument(
        "--use-hybrid",
        action="store_true",
        help="Enable hybrid retrieval (dense + BM25) without reranking",
    )
    parser.add_argument(
        "--use-reranking",
        action="store_true",
        help="Use hybrid retrieval + reranking",
    )
    parser.add_argument(
        "--output",
        default="assistant/results/retrieval_results.jsonl",
        help="Path to JSONL output for per-question retrieval details",
    )
    args = parser.parse_args()

    default_output = Path("assistant/results/retrieval_results.jsonl")
    output_path = Path(args.output)
    if args.output == str(default_output):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = default_output.with_name(f"{default_output.stem}_{timestamp}{default_output.suffix}")

    questions_path = Path(args.questions)
    if not questions_path.exists():
        raise SystemExit(f"Questions file not found: {questions_path}")

    config = load_config()
    metrics, rows = evaluate_retrieval(
        questions_path=questions_path,
        config=config,
        top_k=args.top_k,
        use_hybrid=args.use_hybrid,
        use_reranking=args.use_reranking,
    )

    _write_jsonl(output_path, rows)
    print(f"Results written to {output_path}")

    print("Retrieval evaluation")
    print(f"Questions: {metrics['questions']} (skipped {metrics['skipped']})")
    print(f"Hit@1: {metrics['hit1']:.3f} ({int(metrics['hit1'] * metrics['questions'])}/{metrics['questions']})")
    print(f"Hit@3: {metrics['hit3']:.3f} ({int(metrics['hit3'] * metrics['questions'])}/{metrics['questions']})")
    print(f"MRR:   {metrics['mrr']:.3f}")


if __name__ == "__main__":
    main()
