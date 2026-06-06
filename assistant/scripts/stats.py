from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from assistant.ingestion.stats import ProjectStats, collect_project_stats


def print_project_stats(stats: ProjectStats) -> None:
    corpus = stats.corpus
    benchmarks = stats.benchmarks

    print("\n=== Estadísticas del corpus ===")
    print(f"Fichero de semillas: {corpus.seeds_path}")
    print(f"Directorio HTML: {corpus.html_dir}")
    print(f"Directorio Markdown: {corpus.markdown_dir}")
    print(f"URLs semilla consideradas: {corpus.seed_count}")
    print(f"Páginas HTML descargadas correctamente: {corpus.html_count}")
    print(f"Documentos Markdown finales: {corpus.markdown_count}")
    print(f"Páginas/documentos descartados estimados: {corpus.discarded_estimate}")
    print(f"Tamaño medio de documento Markdown: {corpus.avg_markdown_words} palabras")

    print("\n=== Estadísticas de chunks ===")
    print(f"Archivo de chunks: {corpus.chunks_path}")
    print(f"Chunks generados: {corpus.chunk_count}")
    print(f"Tamaño medio de chunk: {corpus.avg_chunk_words} palabras")

    print("\n=== Estadísticas del dataset QA / benchmarks ===")
    print(f"Carpeta de benchmarks: {benchmarks.benchmarks_dir}")
    print(f"Ficheros detectados: {benchmarks.files_count}")
    print(f"Preguntas totales: {benchmarks.total_questions}")
    print(f"Preguntas con fuente esperada anotada: {benchmarks.questions_with_expected_source}")
    print(f"Preguntas con hechos esperados anotados: {benchmarks.questions_with_expected_facts}")
    print(f"Hechos esperados totales anotados: {benchmarks.total_expected_facts}")


def main() -> None:
    stats = collect_project_stats()
    print_project_stats(stats)


if __name__ == "__main__":
    main()