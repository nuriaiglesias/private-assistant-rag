from __future__ import annotations

import argparse
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from assistant.core.config import load_config
from assistant.rag.pipeline import RagPipeline


def main() -> None:
	examples = """\
Variant mapping examples:

# A: Baseline RAG
python scripts/cli/ask.py "question"

# B: RAG + reranking
python scripts/cli/ask.py "question" --use-reranking

# C: RAG + orchestrator
python scripts/cli/ask.py "question" --use-orchestrator

# D: RAG + reranking + orchestrator
python scripts/cli/ask.py "question" --use-reranking --use-orchestrator
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
	if response.plan:
		print("\nPlan:")
		print(f"- question_type: {response.plan.question_type}")
		print(f"- intent: {response.plan.intent}")
		print(f"- action: {response.plan.action}")
		if response.plan.subqueries:
			print("- subqueries:")
			for index, subquery in enumerate(response.plan.subqueries, start=1):
				print(f"  {index}. {subquery}")
	if response.tool_calls:
		print("- tools_used:")
		for call in response.tool_calls:
			print(f"  - {call.tool_name}")
	if response.email_draft:
		print("\nEmail draft:\n")
		to_value = response.email_draft.to or "[pending]"
		print(f"To: {to_value}")
		print(f"Subject: {response.email_draft.subject}")
		print(response.email_draft.body)
		print("\nEmail status:")
		print("Borrador preparado. El envio requiere confirmacion explicita del usuario.")
	print("\nSources:")
	for index, chunk in enumerate(response.sources, start=1):
		name = Path(chunk.source).name
		print(f"[{index}] {name} (chunk {chunk.chunk_index}, score={chunk.score:.4f})")


if __name__ == "__main__":
	main()
