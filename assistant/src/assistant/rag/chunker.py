from __future__ import annotations

"""Text chunking utilities for document ingestion and retrieval indexing."""

import re
from typing import Iterable, List


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
    min_chunk_size: int = 600,
    max_chunk_size: int = 900,
) -> List[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    if chunk_size < min_chunk_size or chunk_size > max_chunk_size:
        chunk_size = max(min(chunk_size, max_chunk_size), min_chunk_size)

    if overlap >= chunk_size:
        overlap = max(chunk_size // 4, 1)

    units = _split_into_units(normalized, max_chunk_size)
    return _build_chunks(units, chunk_size, overlap)


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _split_into_units(text: str, max_chunk_size: int) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    units: List[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chunk_size:
            units.append(paragraph)
            continue

        units.extend(_split_long_text(paragraph, max_chunk_size))

    return units


def _split_long_text(text: str, max_chunk_size: int) -> List[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) <= 1:
        return _split_by_words(text, max_chunk_size)

    chunks: List[str] = []
    buffer: List[str] = []
    buffer_len = 0
    for sentence in sentences:
        if len(sentence) > max_chunk_size:
            for part in _split_by_words(sentence, max_chunk_size):
                if buffer_len + len(part) + (1 if buffer else 0) > max_chunk_size:
                    chunks.append(" ".join(buffer).strip())
                    buffer = [part]
                    buffer_len = len(part)
                else:
                    buffer.append(part)
                    buffer_len += len(part) + (1 if buffer_len else 0)
            continue

        if buffer_len + len(sentence) + (1 if buffer else 0) > max_chunk_size:
            chunks.append(" ".join(buffer).strip())
            buffer = [sentence]
            buffer_len = len(sentence)
        else:
            buffer.append(sentence)
            buffer_len += len(sentence) + (1 if buffer_len else 0)

    if buffer:
        chunks.append(" ".join(buffer).strip())

    return chunks


def _split_by_words(text: str, max_chunk_size: int) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    buffer: List[str] = []
    buffer_len = 0
    for word in words:
        word_len = len(word) + (1 if buffer else 0)
        if buffer_len + word_len > max_chunk_size:
            chunks.append(" ".join(buffer).strip())
            buffer = [word]
            buffer_len = len(word)
        else:
            buffer.append(word)
            buffer_len += word_len

    if buffer:
        chunks.append(" ".join(buffer).strip())

    return chunks


def _build_chunks(units: List[str], chunk_size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    buffer: List[str] = []
    buffer_len = 0

    for unit in units:
        unit_len = len(unit) + (2 if buffer else 0)
        if buffer_len + unit_len > chunk_size and buffer:
            chunk = "\n\n".join(buffer).strip()
            if chunk:
                chunks.append(chunk)
            buffer, buffer_len = _apply_overlap(buffer, overlap)

        buffer.append(unit)
        buffer_len += unit_len

    if buffer:
        chunk = "\n\n".join(buffer).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def _apply_overlap(buffer: List[str], overlap: int) -> tuple[List[str], int]:
    if overlap <= 0:
        return [], 0

    overlap_units: List[str] = []
    overlap_len = 0
    for unit in reversed(buffer):
        unit_len = len(unit) + (2 if overlap_units else 0)
        if overlap_len + unit_len > overlap and overlap_units:
            break
        overlap_units.insert(0, unit)
        overlap_len += unit_len

    return overlap_units, overlap_len


def chunk_documents(
    documents: Iterable[str],
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[str]:
    all_chunks: List[str] = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
