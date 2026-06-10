# TFM — RAG Assistant for UNIR

Master's Dissertation project in Artificial Intelligence.

## Repository layout

```
assistant/          Python project (RAG assistant)
  corpus/           Knowledge base
    processed_md/   26 Markdown documents (the corpus)
    raw/            Seed URLs and raw data
  data/raw/         Scraped HTML pages
  datasets/         Evaluation question sets (71 Q UNIR, 6 Q email)
  results/          Experiment outputs (JSONL + logs)
  scripts/          Runnable entry points
    ingest/         scrape_unir.py, ingest_documents.py, reset_vector_store.py
    query/          ask.py, search_documents.py
    eval/           evaluate_retrieval.py, evaluate_answers.py,
                    evaluate_orchestrator.py, evaluate_public_dataset.py
  src/assistant/    Python package
    core/           config, paths
    ingestion/      scraper, cleaner, document_loader, ingest
    rag/            chunker, vector_store, retriever, reranker, pipeline
    orchestrator/   router, rules, planner, evidence, orchestrator, state
    llm/            factory, openai_provider (vLLM), ollama_provider
    evaluation/     retrieval, answers, orchestrator, public_dataset
    observability/  Phoenix / OpenTelemetry tracing
    tools/          email_tool, summary_tool, registry
    app/            streamlit_app.py (web UI)
  docker-compose.yml  Qdrant + Phoenix services
  requirements.txt
  requirements-llm.txt  (vLLM CPU — heavy, install only when generating answers)
thesis/             LaTeX sources for the dissertation (TFM.pdf)
planning/           Sprint backlogs, metrics rubric, evaluation dataset CSV
deliveries/         First submission archive
```

## Quick start (from `assistant/`)

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-llm.txt   # only needed to generate answers

# 3. Copy and edit the environment file
cp .env.example .env
# edit .env — set LLM_PROVIDER, LLM_MODEL, and optionally PHOENIX_ENABLED

# 4. Start infrastructure
docker compose up -d   # Qdrant :6333 + Phoenix :6006

# 5. Ingest the corpus into Qdrant
python3 scripts/ingest/ingest_documents.py

# 6. Start the LLM backend (Ollama example)
ollama serve && ollama pull qwen2.5:0.5b-instruct

# 7. Smoke test
python3 scripts/query/ask.py "Que requisitos de acceso pide UNIR?"
```

## Experimental variants (A–D)

| Variant | Command flags | What it tests |
|---------|--------------|---------------|
| A | *(none)* | Baseline: dense retrieval + LLM |
| B | `--use-reranking` | Hybrid (dense+BM25 RRF) + cross-encoder rerank |
| C | `--use-orchestrator` | Agentic: routing, multi-query, evidence gating, email tool |
| D | `--use-reranking --use-orchestrator` | Full system (B + C) |

## Running all evaluations (from `assistant/`)

```bash
# Retrieval quality
python3 scripts/eval/evaluate_retrieval.py
python3 scripts/eval/evaluate_retrieval.py --use-hybrid
python3 scripts/eval/evaluate_retrieval.py --use-reranking

# Answer quality (A–D)
python3 scripts/eval/evaluate_answers.py
python3 scripts/eval/evaluate_answers.py --use-reranking
python3 scripts/eval/evaluate_answers.py --use-orchestrator
python3 scripts/eval/evaluate_answers.py --use-reranking --use-orchestrator

# Orchestrator comparison
python3 scripts/eval/evaluate_orchestrator.py --compare

# External validity (XQuAD-es)
python3 scripts/eval/evaluate_public_dataset.py --mode hybrid --limit 300
```

Results are written to `assistant/results/` as timestamped JSONL files.
Traces are available in Phoenix at `http://localhost:6006`.
