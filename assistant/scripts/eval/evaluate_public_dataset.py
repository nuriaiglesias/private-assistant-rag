"""Evaluate retrieval quality on a public Spanish QA dataset (XQuAD-es)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from assistant.core.config import load_config
from assistant.evaluation.public_dataset import evaluate_public_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval on XQuAD-es (public Spanish QA dataset)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate (default: all ~1190)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Candidate pool size for hybrid/reranking (default: 20)",
    )
    parser.add_argument(
        "--mode",
        choices=["semantic", "hybrid"],
        default="semantic",
        help="Retrieval mode: semantic (dense only) or hybrid (dense + BM25 via RRF)",
    )
    args = parser.parse_args()

    config = load_config()
    metrics = evaluate_public_dataset(
        config=config,
        mode=args.mode,
        limit=args.limit,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
    )

    print("\n" + "=" * 40)
    print("XQuAD-es retrieval evaluation")
    print(f"Mode:          {metrics['mode']}")
    print(f"Embedding:     {config.embedding_model}")
    print(f"Questions:     {metrics['questions']}")
    print(f"Context pool:  {metrics['contexts']} unique paragraphs")
    print(f"Top-k:         {metrics['top_k']}")
    print("=" * 40)
    print(f"Hit@1: {metrics['hit1']:.3f} ({int(metrics['hit1'] * metrics['questions'])}/{metrics['questions']})")
    print(f"Hit@3: {metrics['hit3']:.3f} ({int(metrics['hit3'] * metrics['questions'])}/{metrics['questions']})")
    print(f"MRR:   {metrics['mrr']:.3f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
