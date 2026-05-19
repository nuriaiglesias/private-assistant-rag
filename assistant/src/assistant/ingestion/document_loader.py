from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import fitz


@dataclass(frozen=True)
class Document:
	text: str
	metadata: dict


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def load_documents(root_dir: Path) -> List[Document]:
	documents: List[Document] = []
	for path in sorted(root_dir.rglob("*")):
		if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
			documents.extend(_load_document(path))
	return documents


def _load_document(path: Path) -> Iterable[Document]:
	if path.suffix.lower() == ".pdf":
		text = _load_pdf(path)
	else:
		text = path.read_text(encoding="utf-8", errors="ignore")

	metadata: dict = {}
	if path.suffix.lower() == ".md":
		text, metadata = _parse_frontmatter(text)

	normalized = text.strip()
	if not normalized:
		return []

	return [Document(text=normalized, metadata=_normalize_metadata(metadata, path))]


def _parse_frontmatter(text: str) -> Tuple[str, dict]:
	lines = text.splitlines()
	if not lines or lines[0].strip() != "---":
		return text, {}

	end_index = None
	for index in range(1, len(lines)):
		if lines[index].strip() == "---":
			end_index = index
			break

	if end_index is None:
		return text, {}

	frontmatter = "\n".join(lines[1:end_index]).strip()
	content = "\n".join(lines[end_index + 1 :])
	if not frontmatter:
		return content, {}

	try:
		metadata = json.loads(frontmatter)
		if isinstance(metadata, dict):
			return content, metadata
	except json.JSONDecodeError:
		return text, {}

	return content, {}


def _normalize_metadata(metadata: dict, path: Path) -> dict:
	result = dict(metadata) if metadata else {}
	source_path = result.get("source_path") or str(path)
	result["source_path"] = source_path

	if "source" not in result:
		result["source"] = result.get("source_url", source_path)

	return result


def _load_pdf(path: Path) -> str:
	pages: List[str] = []
	with fitz.open(path) as doc:
		for page in doc:
			pages.append(page.get_text("text"))
	return "\n".join(pages)
