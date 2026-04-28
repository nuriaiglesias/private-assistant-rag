# Sprint 1 - Initial Logical Architecture

Date: 2026-04-27
Status: Draft completed, pending review

This file is the single source for architecture and stack decision (consolidated from ADR-001).

## High-Level Architecture

```mermaid
flowchart TD
    U[User] --> UI[Chainlit Chat UI]
    UI --> ORCH[Orchestrator Agent]

    ORCH --> INTENT[Intent Router (rules first)]
    INTENT --> RETR[Retriever]
    RETR --> VEC[Vector Search]
    RETR --> LEX[Lexical Search (optional)]
    VEC --> RR[Reranker (optional)]
    LEX --> RR
    RR --> HS[Hybrid Scoring (optional)]
    HS --> CTX[Grounded Context Builder]

    CTX --> LLM[LLM Response Generator]
    INTENT --> TOOL_EMAIL[Tool: Email Draft]
    INTENT --> TOOL_SUM[Tool: Summary]

    LLM --> RESP[Answer with citations]
    TOOL_EMAIL --> RESP
    TOOL_SUM --> RESP
    RESP --> UI

    DOCS[Internal Documents] --> ING[Ingestion + Chunking]
    ING --> EMB[Embedding Generation]
    EMB --> VEC
    ING --> LEX

    ORCH --> LOG[Structured Logs]
    RETR --> LOG
    LLM --> LOG
```

## Component Responsibilities

1. UI: Chainlit chat for fast MVP iteration and conversational feel.
2. Orchestrator Agent: Coordinates retrieval, generation, and tools.
3. Intent Router: Rule-based routing for MVP (LLM-based later).
4. Retriever: Baseline semantic retrieval, with optional hybrid mode.
5. Reranker: Optional improvement stage for retrieval quality.
6. Hybrid Scoring (optional): Combine vector score, reranker score, metadata, and diversity.
7. Context Builder: Produces evidence package for LLM.
8. LLM Generator: Produces grounded answer with citations.
9. Tools: Supervised email draft and summary tool.
10. Logging: Captures telemetry for evaluation and debugging.

## End-to-End Query Flow

1. User submits query in Chainlit.
2. Orchestrator classifies intent (informational, action, summary, general).
3. Retriever fetches candidates (baseline semantic; hybrid optional).
4. Optional reranker scores candidates.
5. Optional hybrid scorer combines signals.
6. Context builder creates evidence context.
7. LLM generates grounded answer.
8. Tools generate email draft or summary when requested.
9. UI returns output with citations and metadata.

## Technical Stack Decision (S1-06)

Selected stack for the first implementation iteration (mixed proposal):

1. Language/runtime: Python 3.11.
2. UI: Chainlit (single-app MVP).
3. Backend API: Not required in MVP; FastAPI planned as optional next step.
4. Vector store (local): Qdrant (MVP).
5. Embeddings: BGE-M3 (local) via Sentence Transformers / Hugging Face; OpenAI embeddings optional later.
6. Lexical retrieval (optional): BM25 for hybrid retrieval stage.
7. Reranking (baseline): BGE reranker (bge-reranker-v2-m3).
8. LLM: Ollama for development (local-first MVP) with optional switch to OpenAI API.
    - Base model: qwen3:8b
    - Fallback if resources are tight: llama3.2:3b
9. Tool integration: email draft tool + summary tool (mocked, supervised).
10. Config: .env with python-dotenv for keys and settings.
11. Evaluation tooling: pandas, numpy, scikit-learn, and custom rubric scripts.
12. Deployment: Docker optional for later private deployment.

## Document ingestion and chunking

PDF
-> PyMuPDF
-> page text + metadata
-> chunking by page/section + size

DOCX / HTML / PPTX
-> Pandoc
-> structured Markdown
-> chunking by headings + size

TXT / MD
-> direct read
-> chunking by headings or size

XLSX / CSV
-> tabular extraction
-> chunks by sheet/row/block

Why this stack:

1. Fast MVP with a single Chainlit app.
2. Private, local vector store to keep data in the environment.
3. Clear path to research comparison: baseline vs hybrid + reranking.
4. Compatible with private deployment later (Docker).

Alternatives deferred:

1. Full FastAPI backend from day 1 (kept for later phase).
2. Complex multi-agent system in MVP (kept for future extension).

Main trade-offs:

1. Chainlit MVP sacrifices API separation in exchange for speed.
2. For an easy future UI change, keep assistant logic (RAG, orchestrator, tools) separated and exposed as an API or module; Chainlit should remain only the UI.
3. Local-first LLM may reduce answer quality vs hosted models, but avoids API cost during experimentation.
4. Optional hybrid + reranking adds latency and complexity in later stages.

## Optional research improvements (if time allows)

Fine-tuning an existing reranker

This would start from a model such as bge-reranker-v2-m3 and adapt it with your own examples.

The model bge-reranker-v2-m3 is designed to receive a query and a passage and return a relevance score; unlike an embedding model, the reranker evaluates the query-passage pair directly.

Pros
More realistic than training from scratch.
You can specialize the model to the TFM/university domain.
Technically defendable.
Sentence Transformers supports training or fine-tuning rerankers.

Cons
You still need labeled data.
You must prepare positive and negative pairs.
May require a GPU.
Improvements can be small.
Weak data can degrade the model.
More work than it seems.

Opinion: possible but risky. Only worth it if the thesis focuses heavily on this experimental part.

Hybrid weighted reranker strategy (custom scoring)

You do not train a new model, but you define your own scoring formula:

score_final =
    a * vector_score
+ b * reranker_score
+ c * metadata_score
+ d * diversity_score

Example:

score_final =
    0.25 * qdrant_similarity
+ 0.55 * bge_reranker_score
+ 0.10 * section_match
+ 0.10 * document_diversity

This is a strong original contribution without training a new model.

What you could improve
1. Semantic relevance

Provided by the reranker:

Does this passage actually answer the question?
2. Vector similarity

Provided by Qdrant:

Is this passage close to the question in vector space?
3. Metadata

For example:

document = tfm_regulation
section = evaluation
type = policy
course = 2025/2026
4. Diversity

Avoid the top-3 final chunks being nearly identical from the same section.

5. Low-confidence penalty

If the reranker scores are low across the board, the system can say:

Not enough information found in the available documentation.

This is valuable to reduce hallucinations.
