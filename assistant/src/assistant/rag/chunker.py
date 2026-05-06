from __future__ import annotations

from typing import Iterable, List


def chunk_text(
	text: str,
	chunk_size: int = 800,
	overlap: int = 150,
	min_chunk_size: int = 600,
	max_chunk_size: int = 900,
) -> List[str]:
	if chunk_size < min_chunk_size or chunk_size > max_chunk_size:
		chunk_size = max(min(chunk_size, max_chunk_size), min_chunk_size)

	if overlap >= chunk_size:
		overlap = max(chunk_size // 4, 1)

	chunks: List[str] = []
	start = 0
	length = len(text)

	while start < length:
		end = min(start + chunk_size, length)

		if end < length:
			boundary = text.rfind(" ", start, end)
			if boundary > start + min_chunk_size:
				end = boundary

		chunk = text[start:end].strip()
		if chunk:
			chunks.append(chunk)

		if end == length:
			break

		start = max(end - overlap, 0)

	return chunks


def chunk_documents(
	documents: Iterable[str],
	chunk_size: int = 800,
	overlap: int = 150,
) -> List[str]:
	all_chunks: List[str] = []
	for doc in documents:
		all_chunks.extend(chunk_text(doc, chunk_size=chunk_size, overlap=overlap))
	return all_chunks
