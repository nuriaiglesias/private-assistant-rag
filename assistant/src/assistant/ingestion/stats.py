from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from assistant.core.paths import (
    BENCHMARKS_DIR,
    UNIR_CHUNKS_PATH,
    UNIR_HTML_DIR,
    UNIR_MARKDOWN_DIR,
    UNIR_SEEDS_PATH,
)


BENCHMARK_EXTENSIONS = {".json", ".jsonl", ".csv"}


@dataclass(frozen=True)
class CorpusStats:
    """Resumen estadístico del corpus local utilizado por el sistema RAG.

    Agrupa las métricas relacionadas con los documentos de entrada, los HTML
    descargados, los Markdown procesados y los chunks generados.
    """

    seeds_path: Path
    html_dir: Path
    markdown_dir: Path
    chunks_path: Path
    seed_count: int
    html_count: int
    markdown_count: int
    discarded_estimate: int
    avg_markdown_words: float
    chunk_count: int
    avg_chunk_words: float


@dataclass(frozen=True)
class BenchmarkStats:
    """Resumen estadístico de los benchmarks o datasets de evaluación.

    Agrupa el número de ficheros detectados, preguntas totales y anotaciones
    esperadas utilizadas para evaluar respuestas del sistema.
    """

    benchmarks_dir: Path
    files_count: int
    total_questions: int
    questions_with_expected_source: int
    questions_with_expected_facts: int
    total_expected_facts: int


@dataclass(frozen=True)
class ProjectStats:
    """Resumen global de estadísticas del proyecto.

    Combina las estadísticas del corpus y de los benchmarks en una única
    estructura para facilitar su uso desde scripts u otros módulos.
    """

    corpus: CorpusStats
    benchmarks: BenchmarkStats


def count_words(text: str) -> int:
    """Count the number of words in a text.

    The function uses a regular expression compatible with Unicode characters,
    so it works with Spanish accents and other non-ASCII characters.

    Args:
        text: Text whose words should be counted.

    Returns:
        Number of word-like tokens found in the text.
    """

    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def average(values: list[int]) -> float:
    """Calculate the arithmetic mean of a list of integers.

    If the input list is empty, the function returns 0.0 to avoid division by
    zero. The result is rounded to two decimal places.

    Args:
        values: List of integer values.

    Returns:
        Average value rounded to two decimals, or 0.0 if the list is empty.
    """

    return round(sum(values) / len(values), 2) if values else 0.0


def count_records(path: Path) -> int:
    """Count records in a supported structured file.

    Supported formats are JSON, JSONL and CSV. For JSON files, a list counts as
    multiple records and a dictionary counts as one record. For JSONL files,
    each non-empty line is counted as one record. For CSV files, each data row
    is counted as one record.

    Args:
        path: Path to the file whose records should be counted.

    Returns:
        Number of records found in the file. Returns 0 if the file does not
        exist or if the extension is not supported.
    """

    if not path.exists():
        return 0

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            return sum(1 for line in file if line.strip())

    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, list):
            return len(data)

        if isinstance(data, dict):
            return 1

    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8") as file:
            return sum(1 for _ in csv.DictReader(file))

    return 0


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a structured file and return its rows as dictionaries.

    Supported formats are JSON, JSONL and CSV. JSON files may contain either a
    list of dictionaries or a single dictionary. JSONL files are read line by
    line. CSV files are read using ``csv.DictReader``.

    Args:
        path: Path to the file to read.

    Returns:
        List of rows represented as dictionaries. Returns an empty list if the
        file does not exist, if the extension is unsupported, or if the JSON
        content does not contain dictionaries.
    """

    if not path.exists():
        return []

    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]

        if isinstance(data, dict):
            return [data]

        return []

    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []

        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    item = json.loads(line)
                    if isinstance(item, dict):
                        rows.append(item)

        return rows

    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    return []


def get_chunk_text(chunk: dict[str, Any]) -> str:
    """Extract the text content from a chunk dictionary.

    Different parts of the project or previous versions of the pipeline may use
    different field names for the chunk text. This function keeps compatibility
    by checking the most common alternatives in priority order.

    Args:
        chunk: Dictionary representing a text chunk.

    Returns:
        Text contained in the chunk. Returns an empty string if no compatible
        text field is found.
    """

    return str(
        chunk.get("text")
        or chunk.get("content")
        or chunk.get("page_content")
        or chunk.get("chunk")
        or ""
    )


def count_expected_sources(row: dict[str, Any]) -> int:
    """Count the expected source annotations in a benchmark row.

    The function supports several possible field names to preserve compatibility
    with different dataset versions.

    Args:
        row: Benchmark row represented as a dictionary.

    Returns:
        Number of expected sources. Returns 0 if no source annotation exists.
        If the field is a list, the number of items is returned. Otherwise, a
        non-empty value counts as one source.
    """

    sources = (
        row.get("expected_sources")
        or row.get("expected_source")
        or row.get("sources")
        or row.get("source")
    )

    if not sources:
        return 0

    return len(sources) if isinstance(sources, list) else 1


def count_expected_facts(row: dict[str, Any]) -> int:
    """Count the expected fact annotations in a benchmark row.

    Expected facts may be stored as a list, as a JSON-encoded list, or as a
    semicolon-separated string. This function normalizes those cases into a
    single count.

    Args:
        row: Benchmark row represented as a dictionary.

    Returns:
        Number of expected facts. Returns 0 if the row does not contain expected
        fact annotations.
    """

    facts = (
        row.get("expected_facts")
        or row.get("facts")
        or row.get("expected_answer_facts")
    )

    if not facts:
        return 0

    if isinstance(facts, list):
        return len(facts)

    if isinstance(facts, str):
        try:
            parsed = json.loads(facts)
            if isinstance(parsed, list):
                return len(parsed)
        except json.JSONDecodeError:
            pass

        return len([fact for fact in facts.split(";") if fact.strip()])

    return 1


def collect_corpus_stats() -> CorpusStats:
    """Collect statistics from the local UNIR corpus files.

    The function counts seed URLs, downloaded HTML files, processed Markdown
    documents and generated chunks. It also computes average word counts for
    Markdown documents and chunks.

    Returns:
        CorpusStats object containing all corpus-related statistics.
    """

    html_files = list(UNIR_HTML_DIR.rglob("*.html")) if UNIR_HTML_DIR.exists() else []
    markdown_files = (
        list(UNIR_MARKDOWN_DIR.rglob("*.md"))
        if UNIR_MARKDOWN_DIR.exists()
        else []
    )

    markdown_word_counts = [
        count_words(path.read_text(encoding="utf-8", errors="ignore"))
        for path in markdown_files
    ]

    chunks = read_rows(UNIR_CHUNKS_PATH)
    chunk_word_counts = [count_words(get_chunk_text(chunk)) for chunk in chunks]

    return CorpusStats(
        seeds_path=UNIR_SEEDS_PATH,
        html_dir=UNIR_HTML_DIR,
        markdown_dir=UNIR_MARKDOWN_DIR,
        chunks_path=UNIR_CHUNKS_PATH,
        seed_count=count_records(UNIR_SEEDS_PATH),
        html_count=len(html_files),
        markdown_count=len(markdown_files),
        discarded_estimate=max(len(html_files) - len(markdown_files), 0),
        avg_markdown_words=average(markdown_word_counts),
        chunk_count=len(chunks),
        avg_chunk_words=average(chunk_word_counts),
    )


def collect_benchmark_stats() -> BenchmarkStats:
    """Collect statistics from benchmark and QA dataset files.

    The function scans the benchmarks directory recursively and reads all files
    with supported extensions. It counts total questions, questions with
    expected sources, questions with expected facts and the total number of
    expected facts.

    Returns:
        BenchmarkStats object containing all benchmark-related statistics.
    """

    benchmark_files = (
        sorted(
            path
            for path in BENCHMARKS_DIR.rglob("*")
            if path.is_file() and path.suffix in BENCHMARK_EXTENSIONS
        )
        if BENCHMARKS_DIR.exists()
        else []
    )

    total_questions = 0
    questions_with_expected_source = 0
    questions_with_expected_facts = 0
    total_expected_facts = 0

    for path in benchmark_files:
        rows = read_rows(path)
        total_questions += len(rows)

        for row in rows:
            expected_sources = count_expected_sources(row)
            expected_facts = count_expected_facts(row)

            questions_with_expected_source += int(expected_sources > 0)
            questions_with_expected_facts += int(expected_facts > 0)
            total_expected_facts += expected_facts

    return BenchmarkStats(
        benchmarks_dir=BENCHMARKS_DIR,
        files_count=len(benchmark_files),
        total_questions=total_questions,
        questions_with_expected_source=questions_with_expected_source,
        questions_with_expected_facts=questions_with_expected_facts,
        total_expected_facts=total_expected_facts,
    )


def collect_project_stats() -> ProjectStats:
    """Collect all available project statistics.

    This is the main entry point for statistics collection. It combines corpus
    statistics and benchmark statistics into a single object.

    Returns:
        ProjectStats object containing corpus and benchmark statistics.
    """

    return ProjectStats(
        corpus=collect_corpus_stats(),
        benchmarks=collect_benchmark_stats(),
    )