# Sprint 1 - Baseline Metrics and Evaluation Rubric

Date: 2026-04-27
Status: Draft completed, pending review

## Objective

Define reproducible metrics to compare baseline retrieval and improved retrieval.

## Quantitative Retrieval Metrics

1. Recall@5
Definition: fraction of queries where at least one relevant document appears in top 5.

2. MRR@5
Definition: mean reciprocal rank of first relevant result within top 5.

## Response Quality Rubric (manual)

Each answer is scored from 0 to 2 on each dimension.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Grounding | No evidence or wrong source | Partial evidence | Clear evidence with correct citation |
| Factual accuracy | Major factual errors | Minor inaccuracies | Accurate to source |
| Usefulness | Not actionable / vague | Partially useful | Directly useful for user need |
| Clarity | Hard to understand | Acceptable clarity | Clear and concise |
| Safety/compliance | Unsafe or policy-violating output | Minor issues | Fully aligned and safe |
| Action draft quality (action intents only) | Unusable draft | Needs heavy edits | Ready with minor edits |

## Operational Metrics

1. End-to-end latency (mean, p95).
2. Response success rate.
3. Action draft generation success rate (action subset).

## Acceptance Targets (from Sprint 1 criteria)

1. >= 90% response success rate.
2. >= 90% action draft generation on action queries.
3. >= 75% answers with clear grounding.
4. Improved mode reaches either:
   - +3 percentage points in Recall@5, or
   - +5% relative improvement in MRR@5.
5. Improved mode latency increase <= 50% vs baseline.

## Evaluation Procedure

1. Run baseline mode on fixed dataset.
2. Store retrieval outputs and response logs.
3. Run improved mode on same dataset and settings except retrieval pipeline.
4. Compute quantitative metrics.
5. Perform manual rubric scoring with the same reviewer criteria.
6. Produce comparison table and conclusion.
