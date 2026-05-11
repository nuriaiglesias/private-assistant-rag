from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from assistant.core.config import load_config
from assistant.rag.pipeline import RagPipeline


def main() -> None:
	examples = """\
Variant mapping examples:

# A: Baseline RAG
python scripts/ask.py "question"

# B: RAG + reranking
python scripts/ask.py "question" --use-reranking

# C: RAG + orchestrator
python scripts/ask.py "question" --use-orchestrator

# D: RAG + reranking + orchestrator
python scripts/ask.py "question" --use-reranking --use-orchestrator
"""
	parser = argparse.ArgumentParser(
		description="Ask a question to the RAG pipeline",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog=examples,
	)
	parser.add_argument("question", help="User question")
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
	args = parser.parse_args()

	config = load_config()
	pipeline = RagPipeline(config)
	response = pipeline.run_pipeline(
		args.question,
		top_k=args.top_k,
		use_reranking=args.use_reranking,
		use_orchestrator=args.use_orchestrator,
	)

	print("Answer:\n")
	print(response.answer)
	print("\nSources:")
	for index, chunk in enumerate(response.sources, start=1):
		name = Path(chunk.source).name
		print(f"[{index}] {name} (chunk {chunk.chunk_index}, score={chunk.score:.4f})")


if __name__ == "__main__":
	main()
