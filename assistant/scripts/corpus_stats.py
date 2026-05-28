from pathlib import Path
import json
import re
import csv

REPO_ROOT = Path(__file__).resolve().parents[1]

SEEDS_PATH = REPO_ROOT / "data" / "unir_seeds.jsonl"
RAW_DIR = REPO_ROOT / "data" / "raw" / "unir_html"
MARKDOWN_DIR = REPO_ROOT / "data" / "processed" / "markdown" / "unir"

# Ajusta estos nombres si tu dataset o chunks se llaman distinto
POSSIBLE_CHUNKS = [
    REPO_ROOT / "data" / "processed" / "chunks.jsonl",
    REPO_ROOT / "data" / "processed" / "chunks_unir.jsonl",
    REPO_ROOT / "data" / "processed" / "chunks" / "chunks.jsonl",
]

POSSIBLE_DATASETS = [
    REPO_ROOT / "data" / "evaluation" / "qa_dataset.jsonl",
    REPO_ROOT / "data" / "evaluation" / "qa_dataset.csv",
    REPO_ROOT / "data" / "eval" / "qa_dataset.jsonl",
    REPO_ROOT / "data" / "eval" / "qa_dataset.csv",
]


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text, flags=re.UNICODE))


def find_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def main():
    seed_count = count_jsonl(SEEDS_PATH)

    raw_html_count = len(list(RAW_DIR.rglob("*.html"))) if RAW_DIR.exists() else 0
    markdown_files = list(MARKDOWN_DIR.rglob("*.md")) if MARKDOWN_DIR.exists() else []
    final_docs = len(markdown_files)

    discarded_estimate = max(raw_html_count - final_docs, 0)

    md_word_counts = []
    for path in markdown_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        md_word_counts.append(count_words(text))

    avg_doc_words = round(sum(md_word_counts) / len(md_word_counts), 2) if md_word_counts else 0

    chunks_path = find_existing(POSSIBLE_CHUNKS)
    chunks_count = 0
    avg_chunk_words = 0

    if chunks_path:
        chunks = read_jsonl(chunks_path)
        chunks_count = len(chunks)

        chunk_word_counts = []
        for chunk in chunks:
            text = (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("page_content")
                or chunk.get("chunk")
                or ""
            )
            chunk_word_counts.append(count_words(text))

        avg_chunk_words = round(sum(chunk_word_counts) / len(chunk_word_counts), 2) if chunk_word_counts else 0

    dataset_path = find_existing(POSSIBLE_DATASETS)
    total_questions = 0
    questions_with_expected_source = 0
    questions_with_expected_facts = 0
    total_expected_facts = 0

    if dataset_path:
        if dataset_path.suffix == ".jsonl":
            rows = read_jsonl(dataset_path)
        elif dataset_path.suffix == ".csv":
            with dataset_path.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        else:
            rows = []

        total_questions = len(rows)

        for row in rows:
            expected_source = (
                row.get("expected_source")
                or row.get("expected_sources")
                or row.get("source")
                or row.get("sources")
            )
            if expected_source:
                questions_with_expected_source += 1

            facts = (
                row.get("expected_facts")
                or row.get("facts")
                or row.get("expected_answer_facts")
            )

            if facts:
                questions_with_expected_facts += 1

                if isinstance(facts, list):
                    total_expected_facts += len(facts)
                elif isinstance(facts, str):
                    try:
                        parsed = json.loads(facts)
                        if isinstance(parsed, list):
                            total_expected_facts += len(parsed)
                        else:
                            total_expected_facts += 1
                    except Exception:
                        total_expected_facts += len([x for x in facts.split(";") if x.strip()])
                else:
                    total_expected_facts += 1

    print("\n=== Estadísticas del corpus ===")
    print(f"URLs semilla consideradas: {seed_count}")
    print(f"Páginas HTML descargadas correctamente: {raw_html_count}")
    print(f"Documentos Markdown finales: {final_docs}")
    print(f"Páginas/documentos descartados estimados: {discarded_estimate}")
    print(f"Tamaño medio de documento Markdown: {avg_doc_words} palabras")

    print("\n=== Estadísticas de chunks ===")
    print(f"Archivo de chunks detectado: {chunks_path if chunks_path else 'No detectado'}")
    print(f"Chunks generados: {chunks_count}")
    print(f"Tamaño medio de chunk: {avg_chunk_words} palabras")

    print("\n=== Estadísticas del dataset QA ===")
    print(f"Archivo de dataset detectado: {dataset_path if dataset_path else 'No detectado'}")
    print(f"Preguntas totales: {total_questions}")
    print(f"Preguntas con fuente esperada anotada: {questions_with_expected_source}")
    print(f"Preguntas con hechos esperados anotados: {questions_with_expected_facts}")
    print(f"Hechos esperados totales anotados: {total_expected_facts}")


if __name__ == "__main__":
    main()
