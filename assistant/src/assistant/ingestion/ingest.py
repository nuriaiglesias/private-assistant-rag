from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List

from sentence_transformers import SentenceTransformer

from assistant.core.config import AppConfig
from assistant.ingestion.document_loader import Document, load_documents
from assistant.rag.chunker import chunk_text
from assistant.rag.vector_store import ChunkRecord, QdrantVectorStore


def _get_document_id(doc: Document) -> str:
    """Return a stable document identifier from metadata or content hash."""
    metadata = doc.metadata
    if isinstance(metadata, dict):
        document_id = metadata.get("content_hash") or metadata.get("document_id")
        if document_id:
            return str(document_id)

    return hashlib.sha256(doc.text.encode("utf-8")).hexdigest()


def build_records(documents: Iterable[Document], chunk_size: int, overlap: int) -> List[ChunkRecord]:
    """Convert a stream of documents into vector store chunk records."""
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


def _encode_texts(texts: List[str], embedding_model: str) -> List[List[float]]:
    """Encode a list of text chunks into dense vectors."""
    model = SentenceTransformer(embedding_model)
    return model.encode(texts, show_progress_bar=True).tolist()


def _build_vector_store(config: AppConfig, vector_size: int) -> QdrantVectorStore:
    """Create a Qdrant vector store and ensure the collection exists."""
    store = QdrantVectorStore(
        url=config.qdrant_url,
        collection=config.qdrant_collection,
        vector_size=vector_size,
    )
    store.ensure_collection()
    return store


def ingest_documents(data_root: Path, config: AppConfig) -> int:
    """Ingest documents from disk into Qdrant as vector chunk records."""
    documents = load_documents(data_root)
    if not documents:
        raise ValueError(f"No documents found in {data_root}")

    records = build_records(documents, config.chunk_size, config.chunk_overlap)
    if not records:
        return 0

    vectors = _encode_texts([record.text for record in records], config.embedding_model)
    store = _build_vector_store(config, len(vectors[0]))
    store.upsert(records, vectors)

    return len(records)
