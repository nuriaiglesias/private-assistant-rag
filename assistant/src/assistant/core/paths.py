from __future__ import annotations

import os
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()

    if current.is_file():
        current = current.parent

    for parent in [current, *current.parents]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent

    raise RuntimeError("No se pudo encontrar la raíz del proyecto.")


def env_path(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


REPO_ROOT = find_repo_root()

CORPUS_DIR = env_path("ASSISTANT_CORPUS_DIR", REPO_ROOT / "corpus")
BENCHMARKS_DIR = env_path("ASSISTANT_BENCHMARKS_DIR", REPO_ROOT / "benchmarks")
RESULTS_DIR = env_path("ASSISTANT_RESULTS_DIR", REPO_ROOT / "results")

UNIR_RAW_DIR = CORPUS_DIR / "raw" / "unir"
UNIR_SEEDS_PATH = UNIR_RAW_DIR / "unir_seeds.json"
UNIR_HTML_DIR = UNIR_RAW_DIR / "html"

UNIR_PROCESSED_DIR = CORPUS_DIR / "processed" / "unir"
UNIR_MARKDOWN_DIR = UNIR_PROCESSED_DIR / "markdown"
UNIR_CHUNKS_PATH = UNIR_PROCESSED_DIR / "chunks.jsonl"

PIPELINE_RUNS_PATH = RESULTS_DIR / "pipeline_runs.jsonl"