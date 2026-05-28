from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tqdm import tqdm


@dataclass(frozen=True)
class SeedUrl:
	url: str
	section: str
	category: str
	source_name: str


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "markdown" / "unir"
RAW_DIR = REPO_ROOT / "data" / "raw" / "unir_html"
DEFAULT_SEEDS_PATH = REPO_ROOT / "data" / "unir_seeds.jsonl"

USER_AGENT = "TFM-Nuria-RAG-Corpus/1.0 (+academic use; contact: personal)"
REQUEST_DELAY_SECONDS = 2
SAVE_RAW_HTML = True
SKIP_EXISTING = True
MIN_WORDS = 80

SEED_URLS: List[SeedUrl] = [
	SeedUrl(
		url="https://www.unir.net/estudia-con-nosotros/preguntas-frecuentes/",
		section="FAQs generales",
		category="faqs",
		source_name="unir_public",
	),
	SeedUrl(
		url="https://www.unir.net/estudia-con-nosotros/como-matricularse/",
		section="Matricula",
		category="matricula",
		source_name="unir_public",
	),
	SeedUrl(
		url="https://www.unir.net/universidad-online/normativa/",
		section="Normativa universitaria",
		category="normativa",
		source_name="unir_public",
	),
	SeedUrl(
		url="https://recursosbiblioteca.unir.net/ayudayformacion/preguntasFrecuentes.html",
		section="Biblioteca",
		category="biblioteca",
		source_name="unir_public",
	),
]


def _get_robots_parser(cache: Dict[str, RobotFileParser], url: str) -> RobotFileParser:
	parsed = urlparse(url)
	base = f"{parsed.scheme}://{parsed.netloc}"
	if base in cache:
		return cache[base]

	rp = RobotFileParser()
	rp.set_url(f"{base}/robots.txt")
	try:
		rp.read()
	except Exception:
		pass

	cache[base] = rp
	return rp


def _can_fetch(cache: Dict[str, RobotFileParser], url: str) -> bool:
	rp = _get_robots_parser(cache, url)
	try:
		return rp.can_fetch(USER_AGENT, url)
	except Exception:
		return True


def _clean_html(html: str) -> str:
	soup = BeautifulSoup(html, "html.parser")

	for tag in soup(["script", "style", "nav", "footer", "header", "form", "aside", "noscript"]):
		tag.decompose()

	for tag in soup.find_all(attrs={"role": "navigation"}):
		tag.decompose()

	for tag in soup(["svg"]):
		tag.decompose()

	for img in soup.find_all("img"):
		img.decompose()

	_noise_keywords = (
		"breadcrumb",
		"breadcrumbs",
		"nav",
		"menu",
		"footer",
		"header",
		"cookie",
		"legal",
		"share",
		"social",
		"newsletter",
		"banner",
		"promo",
		"cta",
		"event",
		"events",
		"actualidad",
		"noticia",
		"news",
		"podcast",
		"ranking",
		"rankings",
		"foros",
		"seminario",
	)
	for tag in soup.find_all(True):
		attrs = tag.attrs or {}
		class_value = " ".join(attrs.get("class", [])).lower()
		id_value = (attrs.get("id") or "").lower()
		if any(keyword in class_value for keyword in _noise_keywords):
			tag.decompose()
			continue
		if any(keyword in id_value for keyword in _noise_keywords):
			tag.decompose()

	main = soup.find("main") or soup.find("article") or soup.body or soup
	return str(main)


def _extract_title(html: str) -> str:
	soup = BeautifulSoup(html, "html.parser")
	h1 = soup.find("h1")
	if h1:
		return h1.get_text(" ", strip=True)

	if soup.title:
		return soup.title.get_text(" ", strip=True)

	return "Untitled document"


def _slugify_url(url: str) -> str:
	parsed = urlparse(url)
	slug = parsed.path.strip("/").replace("/", "_") or "home"
	return slug


def _hash_text(text: str) -> str:
	return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _relpath(path: Path) -> str:
	try:
		return path.relative_to(REPO_ROOT).as_posix()
	except ValueError:
		return path.as_posix()


def _build_metadata(seed: SeedUrl, url: str, title: str, markdown: str, output_path: Path) -> dict:
	parsed = urlparse(url)
	return {
		"source_url": url,
		"title": title,
		"section": seed.section,
		"doc_type": "html",
		"domain": parsed.netloc,
		"retrieved_at": time.strftime("%Y-%m-%d"),
		"language": "es",
		"content_hash": _hash_text(markdown),
		"category": seed.category,
		"source_name": seed.source_name,
		"source_path": _relpath(output_path),
	}


def _save_markdown(output_path: Path, metadata: dict, markdown: str) -> None:
	content = "---\n" + json.dumps(metadata, ensure_ascii=False, indent=2) + "\n---\n\n" + markdown
	output_path.write_text(content, encoding="utf-8")


def _save_raw_html(raw_path: Path, html: str) -> None:
	raw_path.write_text(html, encoding="utf-8")


def _output_paths(seed: SeedUrl, url: str) -> tuple[Path, Path]:
	slug = _slugify_url(url)
	digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
	file_name = f"{slug}_{digest}"
	output_path = OUTPUT_DIR / f"{file_name}.md"
	raw_path = RAW_DIR / f"{file_name}.html"
	return output_path, raw_path


def _scrape_one(client: httpx.Client, cache: Dict[str, RobotFileParser], seed: SeedUrl) -> Optional[Path]:
	url = seed.url
	if not _can_fetch(cache, url):
		print(f"Skipped by robots.txt: {url}")
		return None

	output_path, raw_path = _output_paths(seed, url)
	if SKIP_EXISTING and output_path.exists():
		return output_path

	try:
		response = client.get(url, timeout=30)
		response.raise_for_status()
	except httpx.HTTPError as exc:
		print(f"Failed to fetch {url}: {exc}")
		return None

	html = response.text
	if SAVE_RAW_HTML:
		RAW_DIR.mkdir(parents=True, exist_ok=True)
		_save_raw_html(raw_path, html)

	title = _extract_title(html)
	cleaned = _clean_html(html)
	markdown = md(cleaned, heading_style="ATX")
	if len(markdown.split()) < MIN_WORDS:
		print(f"Skipped because content is too short: {url}")
		return None

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	metadata = _build_metadata(seed, url, title, markdown, output_path)
	_save_markdown(output_path, metadata, markdown)
	return output_path


def scrape(seed_urls: Iterable[SeedUrl]) -> List[Path]:
	headers = {"User-Agent": USER_AGENT}
	robots_cache: Dict[str, RobotFileParser] = {}
	outputs: List[Path] = []

	with httpx.Client(headers=headers, follow_redirects=True) as client:
		for seed in tqdm(list(seed_urls)):
			output = _scrape_one(client, robots_cache, seed)
			if output is not None:
				outputs.append(output)
			time.sleep(REQUEST_DELAY_SECONDS)

	return outputs


def _load_seeds(path: Path) -> List[SeedUrl]:
	if not path.exists():
		return SEED_URLS

	seeds: List[SeedUrl] = []
	for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
		stripped = line.strip()
		if not stripped:
			continue

		try:
			payload = json.loads(stripped)
		except json.JSONDecodeError as exc:
			raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc

		if not isinstance(payload, dict):
			raise ValueError(f"Seed on line {line_number} must be a JSON object")

		missing = {"url", "section", "category", "source_name"} - set(payload)
		if missing:
			raise ValueError(f"Seed on line {line_number} missing fields: {sorted(missing)}")

		seeds.append(
			SeedUrl(
				url=payload["url"],
				section=payload["section"],
				category=payload["category"],
				source_name=payload["source_name"],
			)
		)

	return seeds


def main() -> None:
	import argparse

	parser = argparse.ArgumentParser(description="Scrape UNIR public pages from seed URLs")
	parser.add_argument(
		"--seeds",
		default=str(DEFAULT_SEEDS_PATH),
		help="Path to a JSONL file with seed URLs",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Optional limit of seeds to process",
	)
	args = parser.parse_args()

	seed_path = Path(args.seeds)
	seed_urls = _load_seeds(seed_path)
	if args.limit is not None:
		seed_urls = seed_urls[: args.limit]

	outputs = scrape(seed_urls)
	print(f"Saved {len(outputs)} documents to {OUTPUT_DIR}")


if __name__ == "__main__":
	main()
