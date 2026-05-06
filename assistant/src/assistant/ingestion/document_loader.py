from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

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

	normalized = text.strip()
	if not normalized:
		return []

	return [Document(text=normalized, metadata={"source": str(path)})]


def _load_pdf(path: Path) -> str:
	pages: List[str] = []
	with fitz.open(path) as doc:
		for page in doc:
			pages.append(page.get_text("text"))
	return "\n".join(pages)
