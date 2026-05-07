from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sentence_transformers import SentenceTransformer

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from assistant.core.config import load_config
from assistant.rag.vector_store import QdrantVectorStore


def _format_result(payload: dict, score: float, index: int) -> str:
	text = payload.get("text", "")
	source = payload.get("source", "unknown")
	chunk_index = payload.get("chunk_index", "?")
	preview = " ".join(text.split())[:240]
	return (
		f"{index}. score={score:.4f}\n"
		f"   source={source}\n"
		f"   chunk_index={chunk_index}\n"
		f"   text={preview}\n"
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Semantic search over Qdrant chunks")
	parser.add_argument("query", help="Search query text")
	parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
	args = parser.parse_args()

	config = load_config()
	model = SentenceTransformer(config.embedding_model)
	query_vector = model.encode([args.query])[0].tolist()

	store = QdrantVectorStore(
		url=config.qdrant_url,
		collection=config.qdrant_collection,
		vector_size=len(query_vector),
	)
	results = store.search(query_vector, limit=args.top_k)

	if not results:
		print("No results found.")
		return

	for index, result in enumerate(results, start=1):
		payload = result.payload or {}
		print(_format_result(payload, result.score, index))


if __name__ == "__main__":
	main()
