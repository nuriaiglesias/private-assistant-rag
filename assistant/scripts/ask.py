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
	parser = argparse.ArgumentParser(description="Ask a question to the RAG pipeline")
	parser.add_argument("question", help="User question")
	parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
	args = parser.parse_args()

	config = load_config()
	pipeline = RagPipeline(config)
	response = pipeline.answer(args.question, top_k=args.top_k)

	print("Answer:\n")
	print(response.answer)
	print("\nSources:")
	for index, chunk in enumerate(response.sources, start=1):
		name = Path(chunk.source).name
		print(f"[{index}] {name} (chunk {chunk.chunk_index}, score={chunk.score:.4f})")


if __name__ == "__main__":
	main()
