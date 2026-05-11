# Experimental design (RAG variants)

## Goal
Define what is compared, why it is compared, and which components are added in each variant.

## Variants (A-D)

| Variant | Retriever | Reranking | Orchestrator | Objective |
| --- | --- | --- | --- | --- |
| A | Yes | No | No | Simple baseline |
| B | Yes | Yes | No | Measure the impact of reranking |
| C | Yes | No | Yes | Measure the impact of orchestration |
| D | Yes | Yes | Yes | Final proposed system |

## What each variant adds

- A: Base RAG with a retriever and a single-pass answer generation.
- B: Adds a reranking step to improve top-k relevance before generation.
- C: Adds an orchestrator (single agent, ReAct style) to plan tool usage.
- D: Combines reranking + orchestrator for the full system.

## Evaluation protocol

All variants will be evaluated using the same corpus, the same set of questions, and the same base LLM configuration. This ensures that the differences observed are mainly caused by the added components: reranking and/or orchestration.

For each variant, the following aspects will be analyzed:

- Retrieval quality: relevance of the retrieved chunks.
- Answer quality: correctness, completeness and grounding in the provided context.
- Traceability: whether the answer includes useful references to the retrieved sources.
- Execution complexity: number of steps, tools used and additional latency introduced.

## Orchestrator tools

The orchestrator is implemented as a single agent following a ReAct-inspired pattern. Its role is to decide which tool should be used at each step depending on the query and the selected system variant.

Available tools:

- `retrieve_chunks`: Query the vector database and return the most relevant chunks.
- `rerank_chunks`: Reorder the retrieved chunks according to their relevance. This tool is only used in variants B and D.
- `answer_with_context`: Generate the final answer using the selected context chunks.
- `summarize_context`: Compress long retrieved contexts when the amount of information is too large for the final generation step.

Tool usage by variant:

- Variant A: `retrieve_chunks` -> `answer_with_context`
- Variant B: `retrieve_chunks` -> `rerank_chunks` -> `answer_with_context`
- Variant C: orchestrator decides between `retrieve_chunks`, `summarize_context` and `answer_with_context`
- Variant D: orchestrator can use the full tool set, including reranking

## Architecture diagram (draft)

```mermaid
flowchart LR
  Q[User question] --> R[Retriever]
  R -->|top-k| K[Context chunks]
  K -->|optional| RR[Reranker]
  RR -->|top-k| K2[Ranked chunks]
  K -->|no rerank| G[Generator]
  K2 --> G

  subgraph Orchestrator (single agent)
    T[Plan (ReAct)] --> A1[retrieve_chunks]
    A1 --> A2[rerank_chunks]
    A2 --> A3[answer_with_context]
  end

  Q --> T
  A3 --> G
  G --> O[Answer]
```
