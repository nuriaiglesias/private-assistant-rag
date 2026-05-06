from __future__ import annotations

from pathlib import Path
import sys
from typing import List

from sentence_transformers import SentenceTransformer

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(SRC_ROOT))

from assistant.core.config import load_config
from assistant.ingestion.document_loader import Document, load_documents
from assistant.rag.chunker import chunk_text
from assistant.rag.vector_store import ChunkRecord, QdrantVectorStore


def _build_records(documents: List[Document], chunk_size: int, overlap: int) -> List[ChunkRecord]:
	records: List[ChunkRecord] = []
	for doc in documents:
		chunks = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)
		for index, chunk in enumerate(chunks):
			metadata = dict(doc.metadata)
			metadata["chunk_index"] = index
			records.append(ChunkRecord(text=chunk, metadata=metadata))
	return records


def main() -> None:
	# Ensure dependencies are installed: pip install -r requirements.txt
	# LLM runtime deps live in requirements-llm.txt and are optional for ingestion.
	config = load_config()
	data_root = Path(__file__).resolve().parents[1] / "data" / "raw"
	documents = load_documents(data_root)

	if not documents:
		raise SystemExit("No documents found in assistant/data/raw")

	records = _build_records(documents, config.chunk_size, config.chunk_overlap)

	model = SentenceTransformer(config.embedding_model)
	vectors = model.encode([record.text for record in records], show_progress_bar=True)

	store = QdrantVectorStore(
		url=config.qdrant_url,
		collection=config.qdrant_collection,
		vector_size=vectors.shape[1],
	)
	store.ensure_collection()
	store.upsert(records, vectors.tolist())

	print(f"Indexed {len(records)} chunks into {config.qdrant_collection}")


if __name__ == "__main__":
	main()
