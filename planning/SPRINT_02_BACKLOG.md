# Sprint 2 - Technical Implementation Backlog

Suggested duration: 10 working days
Status: Ready to start after Sprint 1 review

## Sprint Goal

Deliver a runnable baseline assistant with private RAG, source citations, logging, and an initial supervised email-draft tool.

## Definition of Done (Sprint 2)

1. Project bootstrapped with one-command startup.
2. Health endpoint and chat endpoint operational.
3. Baseline retrieval pipeline operational on curated corpus.
4. Answers include source references.
5. Structured logs capture retrieval and response metadata.
6. Initial action tool generates supervised email drafts.

## Backlog

| ID | Task | Priority | Estimate | Dependencies | Acceptance Criteria |
|---|---|---|---|---|---|
| S2-01 | Create project folder structure | Must | 1.5h | None | app, rag, ingestion, prompts, eval, tests, data, scripts created |
| S2-02 | Configure Python env and dependencies | Must | 2h | S2-01 | Reproducible environment setup document and lock file |
| S2-03 | Implement config management | Must | 1.5h | S2-02 | Single config file controls model/retrieval settings |
| S2-04 | Create FastAPI skeleton (health/chat) | Must | 2h | S2-02 | /health and /chat endpoints return valid responses |
| S2-05 | Implement document loader and cleaner | Must | 2h | S2-01 | Loader produces normalized text and metadata |
| S2-06 | Implement chunking pipeline | Must | 2h | S2-05 | Chunk size/overlap configurable and logged |
| S2-07 | Implement embedding + vector indexing | Must | 3h | S2-06 | Documents indexed in vector DB with chunk IDs |
| S2-08 | Implement baseline retriever (semantic top-k) | Must | 2h | S2-07 | Top-k retrieval returns chunk IDs and scores |
| S2-09 | Implement grounded answer generation with citations | Must | 3h | S2-04,S2-08 | Answer includes cited chunk references |
| S2-10 | Add anti-hallucination prompt rules | Must | 1.5h | S2-09 | Model asks clarification when evidence is weak |
| S2-11 | Add structured logging and trace IDs | Must | 2h | S2-04,S2-08,S2-09 | Logs contain query, retrieved chunks, and output metadata |
| S2-12 | Implement initial action tool (email draft) | Should | 2h | S2-09 | Action intent returns formal draft template |
| S2-13 | Add manual test script for 20-question dataset | Must | 2h | S2-09 | Script runs dataset and stores outputs |
| S2-14 | Build baseline metrics report script | Must | 2h | S2-13 | Outputs latency and basic quality summary |
| S2-15 | Sprint demo and technical notes | Should | 1h | S2-14 | Demo checklist and notes prepared |
