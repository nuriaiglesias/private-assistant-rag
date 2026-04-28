# Sprint 1 - Problem Analysis and Definition

Suggested duration: 5 days
Status: Draft completed, pending review

## Sprint Goal

Define the V1 scope with precision, close key technical decisions, and leave a solid base ready for implementation.

## Definition of Done (Sprint)

1. V1 scope approved with included/excluded items.
2. Initial architecture defined with end-to-end flow.
3. Technical stack selected and justified.
4. Initial evaluation dataset created.
5. Baseline metrics and rubric defined.
6. Main risks identified with mitigations.

## Real Use Cases (S1-01)

| ID | Actor | Need | Assistant outcome | Action required |
|---|---|---|---|---|
| UC-01 | Student | Understand scholarship eligibility | Returns eligibility rules with source citations and a short decision checklist | No |
| UC-02 | Student | Know internship agreement process | Explains ordered steps, required forms, deadlines, and office contact | No |
| UC-03 | Student | Confirm enrollment deadlines and penalties | Provides exact dates, conditions, and official regulation reference | No |
| UC-04 | Student | Resolve timetable conflict | Identifies the formal request process and needed evidence | No |
| UC-05 | Student | Ask exam review procedure | Provides who to contact, timing, and appeal path with references | No |
| UC-06 | Student | Ask recognition/credit transfer rules | Summarizes requirements, constraints, and links to policy sections | No |
| UC-07 | Student | Send a formal request email to academic office | Produces a draft email based on recovered policy and user context | Yes, human review before sending |
| UC-08 | Student | Send a follow-up email to internship tutor | Produces a structured draft with neutral tone and clear subject | Yes, human review before sending |

## V1 Scope (S1-02)

In scope:

1. Answer institutional queries using private internal documents.
2. Provide source references for factual answers.
3. Support short conversational context in session.
4. Use anti-hallucination behavior when evidence is weak.
5. Generate supervised formal email drafts for action intents.
6. Keep retrieval and response traces for debugging and evaluation.
7. Provide a summary tool (mocked) in the MVP.

Out of scope:

1. Automatic email sending without explicit human confirmation.
2. Direct integration with SIS/ERP.
3. Production-grade SLA and high availability.
4. Voice, mobile, or multimodal interfaces.

## Requirements (S1-03 and S1-04)

Functional (must):

1. Accept natural language institutional queries.
2. Retrieve relevant chunks before generation.
3. Include citations in factual answers.
4. Ask for clarification when evidence is insufficient.
5. Expose chat and health endpoints.
6. Log query, retrieval candidates, selected evidence, and output metadata.
7. Support baseline and improved retrieval modes.
8. Generate supervised formal email drafts for action-oriented intents.
9. Provide a summary tool for document-based requests.

Non-functional targets:

1. Baseline mean latency <= 6s per query.
2. Improved retrieval latency increase <= 50% vs baseline.
3. >= 75% answers explicitly grounded in evidence.
4. >= 90% response success rate.
5. >= 90% action-draft success rate on action queries.
6. 100% response traceability (query ID + source references).

## Corpus and Quality Rules (S1-07)

Initial corpus categories:

1. Enrollment regulations and deadlines.
2. Scholarship and financial aid rules.
3. Internship procedures and forms.
4. Academic calendar and exam regulations.
5. Credit recognition/transfer policies.
6. Student service contact protocols.

Processing rules:

1. Normalize to UTF-8 text/markdown.
2. Remove repeated headers/footers.
3. Preserve section titles.
4. Keep dates and numeric identifiers intact.
5. Attach metadata per chunk.

Chunking defaults:

1. Chunk size: 600-900 chars.
2. Overlap: 100-150 chars.

## Risks and Mitigations (S1-11)

| ID | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| R-01 | Integration complexity across LLM, retrieval, and tools | Medium | High | Modular interfaces and stepwise integration |
| R-02 | Corpus quality is insufficient | Medium | High | Early corpus curation with quality checks |
| R-03 | Improved retrieval does not outperform baseline | Medium | Medium | Controlled comparison and objective reporting |
| R-04 | Latency increase from hybrid + reranking | High | Medium | Limit candidates and tune reranker usage |
| R-05 | Scope creep | High | Medium | Enforce V1 boundaries and Must priorities |
| R-06 | Schedule delays | Medium | High | Short milestones and daily tracking |
| R-07 | Single-person workload bottlenecks | High | Medium | Timebox tasks and reduce non-critical scope |

## Architecture and Stack (S1-05 and S1-06)

Architecture summary:

1. Chainlit UI for MVP.
2. Orchestrator with rule-based intent routing.
3. RAG over private documents (baseline semantic retrieval).
4. Optional hybrid retrieval + reranking as improvement stage.
5. Tools: email draft + summary (supervised, mocked).
6. Direct LLM answer for general questions.

Stack summary:

1. Python 3.11.
2. Chainlit UI (MVP).
3. Qdrant local vector store.
5. Embeddings: BGE-M3 (local) via Sentence Transformers / Hugging Face; OpenAI embeddings optional later.
6. Reranker: BGE reranker (bge-reranker-v2-m3).
7. LLM: Ollama for development (local-first MVP) with optional switch to OpenAI API.
	- Base model: qwen3:8b
	- Fallback if resources are tight: llama3.2:3b
   
8. Optional BM25 + reranking for improvement stage.
9. Config via .env and python-dotenv.
10. Docker optional for later private deployment.
11. For an easy future UI change, keep assistant logic (RAG, orchestrator, tools) separated and exposed as an API or module; Chainlit should remain only the UI.

Stack line (compact):

Python + Chainlit + Qdrant + BGE-M3 + BGE reranker + weighted hybrid reranking strategy (optional) + optional reranker fine-tuning + Ollama/OpenAI

## Document ingestion pipeline

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

## Optional research improvements (if time allows)

Opcion B: fine-tuning de un reranker existente

Esto seria partir de un modelo como bge-reranker-v2-m3 y ajustarlo con ejemplos tuyos.

El modelo bge-reranker-v2-m3 ya esta pensado para recibir una consulta y un fragmento, y devolver una puntuacion de relevancia; la propia ficha explica que, a diferencia de un embedding model, el reranker evalua directamente el par query-passage.

Pros
Mas realista que entrenar desde cero.
Puedes especializar el modelo al dominio TFM/universidad.
Es tecnicamente defendible.
Sentence Transformers permite entrenar o fine-tunear modelos de embeddings y rerankers.

Contras
Sigues necesitando datos etiquetados.
Tienes que preparar pares positivos y negativos.
Puede requerir GPU.
La mejora puede ser pequena.
Si el dataset es pobre, puedes empeorar el modelo.
Es mas trabajo del que parece.

Mi opinion: posible, pero arriesgado. Solo lo haria si quieres que tu TFM se centre mucho en esa parte experimental.

Opcion D: reranker hibrido propio

Esta puede ser la opcion mas bonita para ti.

No entrenas un modelo nuevo, pero si defines una formula propia de puntuacion:

score_final =
	a * score_vectorial
+ b * score_reranker
+ c * score_metadatos
+ d * score_diversidad

Por ejemplo:

score_final =
	0.25 * similitud_qdrant
+ 0.55 * score_bge_reranker
+ 0.10 * coincidencia_seccion
+ 0.10 * diversidad_documental

Esto si tiene pinta de aportacion propia.

Que podrias mejorar
1. Relevancia semantica

La da el reranker:

Este fragmento responde realmente a la pregunta?
2. Similitud vectorial

La da Qdrant:

Este fragmento esta cerca de la pregunta en el espacio vectorial?
3. Metadatos

Por ejemplo:

documento = normativa_tfm
seccion = evaluacion
tipo = reglamento
curso = 2025/2026
4. Diversidad

Evitar que el top-3 final sean tres chunks casi iguales del mismo apartado.

5. Penalizacion por baja confianza

Si el reranker da puntuaciones bajas a todo, el sistema puede decir:

No he encontrado informacion suficiente en la documentacion disponible.

Esto ultimo es muy valioso para reducir alucinaciones.

## Operational Board

### To Do

None.

### In Progress

None.

### Done

| ID | Date | Evidence |
|---|---|---|
| S1-01 | 2026-04-27 | Section in this file |
| S1-02 | 2026-04-27 | Section in this file |
| S1-03 | 2026-04-27 | Section in this file |
| S1-04 | 2026-04-27 | Section in this file |
| S1-05 | 2026-04-27 | S1-05_LOGICAL_ARCHITECTURE.md |
| S1-06 | 2026-04-27 | S1-05_LOGICAL_ARCHITECTURE.md (stack section) |
| S1-07 | 2026-04-27 | Section in this file |
| S1-08 | 2026-04-27 | S1-08_EVALUATION_DATASET.csv |
| S1-09 | 2026-04-27 | S1-09_BASELINE_METRICS_RUBRIC.md |
| S1-10 | 2026-04-27 | ../SPRINT_02_BACKLOG.md |
| S1-11 | 2026-04-27 | Section in this file |
| S1-12 | 2026-04-27 | Consolidated in this file |
