from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from assistant.core.config import load_config
from assistant.evaluation.orchestrator import evaluate_orchestrator
from assistant.rag.pipeline import RagPipeline


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate orchestrator routing and tool usage")
    parser.add_argument(
        "--questions",
        default="assistant/datasets/unir_questions.json",
        help="Path to unir_questions.json",
    )
    parser.add_argument(
        "--email-questions",
        default="assistant/datasets/email_questions.json",
        help="Path to email_questions.json",
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
        default="assistant/results/orchestrator_results.jsonl",
        help="Path to JSONL output",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions")
    args = parser.parse_args()

    default_output = Path("assistant/results/orchestrator_results.jsonl")
    output_path = Path(args.output)
    if args.output == str(default_output):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = default_output.with_name(f"{default_output.stem}_{timestamp}{default_output.suffix}")

    questions_path = Path(args.questions)
    if not questions_path.exists():
        raise SystemExit(f"Questions file not found: {questions_path}")

    email_path = Path(args.email_questions)
    if not email_path.exists():
        raise SystemExit(f"Email questions file not found: {email_path}")

    config = load_config()
    pipeline = RagPipeline(config)
    metrics_rows, result_rows, results_by_mode = evaluate_orchestrator(
        questions_path=questions_path,
        email_questions_path=email_path,
        pipeline=pipeline,
        top_k=args.top_k,
        use_hybrid=args.use_hybrid,
        use_reranking=args.use_reranking,
        compare=args.compare,
        mode=args.mode,
    )

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
        if args.mode and output_path == default_output:
            output_path = Path(
                f"data/evaluation/results/orchestrator_results_{args.mode}_{timestamp}.jsonl"
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
