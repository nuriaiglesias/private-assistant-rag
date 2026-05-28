# Development

This folder is reserved for technical implementation.

## Suggested Initial Structure

- `app/` API and core application logic.
- `rag/` ingestion, chunking, retrieval, and reranking.
- `prompts/` system prompts and templates (RAG prompts live in `prompts/rag`).
- `eval/` baseline and comparison evaluation scripts.
- `tests/` unit and integration tests.
- `data/` test documents and evaluation datasets.
- `scripts/` automation utilities.

## Simple Rule

Only code and technical artifacts should live in this folder.

## Phase 1: Environment Setup

Prerequisites:

- Docker and Docker Compose installed.
- Python 3.10+ with `pip`.

Start Qdrant:

```bash
docker compose up -d
```

Install dependencies:

```bash
pip install -r requirements.txt
```

LLM runtime dependencies (heavy):

```bash
pip install -r requirements-llm.txt
```

Run ingestion:

```bash
python3 scripts/ingest/ingest_documents.py
```

If you run scripts from the repository root, set the module path:

```bash
PYTHONPATH=assistant/src python3 assistant/scripts/ingest/ingest_documents.py
```

If you prefer `python`, install the shim:

```bash
sudo apt install python-is-python3
```

## Current Technical Decisions

- UI: Streamlit (simple, decoupled UI layer).
- Vector store: Qdrant.
- LLM runtime: vLLM with a small quantized local model for fast iteration.
- Embeddings: BGE-M3.
- Email tool: mocked for the initial version.
