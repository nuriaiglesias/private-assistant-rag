from __future__ import annotations

import argparse
import hashlib
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
		document_id = _get_document_id(doc)
		chunks = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)
		for index, chunk in enumerate(chunks):
			metadata = dict(doc.metadata)
			metadata["chunk_index"] = index
			metadata["chunk_id"] = f"{document_id}:{index}"
			metadata["document_id"] = document_id
			if "source_url" in metadata and "url" not in metadata:
				metadata["url"] = metadata["source_url"]
			records.append(ChunkRecord(text=chunk, metadata=metadata))
	return records


def _get_document_id(doc: Document) -> str:
	metadata = doc.metadata
	if isinstance(metadata, dict):
		document_id = metadata.get("content_hash") or metadata.get("document_id")
		if document_id:
			return str(document_id)

	return hashlib.sha256(doc.text.encode("utf-8")).hexdigest()


def main() -> None:
	# Ensure dependencies are installed: pip install -r requirements.txt
	# LLM runtime deps live in requirements-llm.txt and are optional for ingestion.
	parser = argparse.ArgumentParser(description="Ingest documents into Qdrant")
	parser.add_argument(
		"--input",
		help="Directory with documents to ingest",
		default="data/raw",
	)
	args = parser.parse_args()

	config = load_config()
	repo_root = Path(__file__).resolve().parents[1]
	input_path = Path(args.input)
	data_root = input_path if input_path.is_absolute() else repo_root / input_path
	data_root = data_root.resolve()
	documents = load_documents(data_root)

	if not documents:
		raise SystemExit(f"No documents found in {data_root}")

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
