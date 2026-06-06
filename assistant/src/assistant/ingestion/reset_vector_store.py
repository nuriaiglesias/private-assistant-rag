from __future__ import annotations

from qdrant_client import QdrantClient

from assistant.core.config import AppConfig


def reset_vector_store(config: AppConfig) -> bool:
    """Delete the configured Qdrant collection if it exists."""
    client = QdrantClient(url=config.qdrant_url)
    existing = [collection.name for collection in client.get_collections().collections]
    if config.qdrant_collection not in existing:
        return False

    client.delete_collection(collection_name=config.qdrant_collection)
    return True
