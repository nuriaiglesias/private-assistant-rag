from __future__ import annotations

"""Qdrant vector store adapter used by the retrieval layer."""

from dataclasses import dataclass
from typing import Iterable, List
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models


@dataclass(frozen=True)
class ChunkRecord:
    text: str
    metadata: dict


class QdrantVectorStore:
    def __init__(self, url: str, collection: str, vector_size: int) -> None:
        self._client = QdrantClient(url=url)
        self._collection = collection
        self._vector_size = vector_size

    def ensure_collection(self) -> None:
        existing = self._client.get_collections().collections
        if any(collection.name == self._collection for collection in existing):
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qdrant_models.VectorParams(
                size=self._vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    def upsert(self, records: Iterable[ChunkRecord], vectors: List[List[float]]) -> None:
        payloads = [record.metadata | {"text": record.text} for record in records]
        points = [
            qdrant_models.PointStruct(id=str(uuid4()), vector=vector, payload=payload)
            for vector, payload in zip(vectors, payloads)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, vector: List[float], limit: int = 5) -> List[qdrant_models.ScoredPoint]:
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return response.points
