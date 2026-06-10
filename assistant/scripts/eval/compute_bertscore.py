"""Post-process answer JSONL files to add BERTScore F1.

Run AFTER evaluate_answers.py --skip-bertscore so the embedding model is no
longer in memory. This script loads only bert-score (no sentence-transformers).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _compute_bertscore_batch(
    answers: list[str], facts_list: list[list[str]]
) -> list[float | None]:
    from bert_score import score as _bert_score_fn

    flat_hyps: list[str] = []
    flat_refs: list[str] = []
    index_map: list[tuple[int, int]] = []  # (row_idx, fact_idx)

    for row_idx, (answer, facts) in enumerate(zip(answers, facts_list)):
        for fact_idx, fact in enumerate(facts):
            flat_hyps.append(answer)
            flat_refs.append(fact)
            index_map.append((row_idx, fact_idx))

    if not flat_hyps:
        return [None] * len(answers)

    _, _, f1 = _bert_score_fn(flat_hyps, flat_refs, lang="es", verbose=False)
    f1_list = f1.tolist()

    # Average per row
    row_scores: list[list[float]] = [[] for _ in range(len(answers))]
    for (row_idx, _), score in zip(index_map, f1_list):
        row_scores[row_idx].append(score)

    return [
        sum(scores) / len(scores) if scores else None
        for scores in row_scores
    ]


def process_file(path: Path, dry_run: bool = False) -> dict:
    rows = _load_jsonl(path)

    answers = []
    facts_list = []
    eligible_indices = []

    for i, row in enumerate(rows):
        if not row.get("answerable", True):
            continue
        if row.get("bertscore_f1") is not None:
            continue  # already computed
        answer = row.get("answer", "")
        facts = row.get("expected_facts") or []
        if not facts or not answer:
            continue
        answers.append(answer)
        facts_list.append(facts)
        eligible_indices.append(i)

    if not eligible_indices:
        print(f"  {path.name}: nothing to compute (already done or no facts)")
        return {"file": str(path), "updated": 0}

    print(f"  {path.name}: computing BERTScore for {len(eligible_indices)} rows...")
    scores = _compute_bertscore_batch(answers, facts_list)

    for idx, score in zip(eligible_indices, scores):
        rows[idx]["bertscore_f1"] = score

    avg = None
    valid = [s for s in scores if s is not None]
    if valid:
        avg = sum(valid) / len(valid)
        print(f"  {path.name}: avg BERTScore F1 = {avg:.4f}")

    if not dry_run:
        _write_jsonl(path, rows)
        print(f"  {path.name}: saved.")

    return {"file": str(path), "updated": len(eligible_indices), "avg_bertscore": avg}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BERTScore for saved answer JSONL files")
    parser.add_argument(
        "--results-dir",
        default="assistant/results",
        help="Directory containing answers_results_*.jsonl files",
    )
    parser.add_argument(
        "--pattern",
        default="answers_results_*.jsonl",
        help="Glob pattern for files to process",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without writing")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    files = sorted(results_dir.glob(args.pattern))

    if not files:
        print(f"No files matching '{args.pattern}' in {results_dir}")
        return

    print(f"Found {len(files)} file(s) to process.")
    summary = []
    for f in files:
        result = process_file(f, dry_run=args.dry_run)
        summary.append(result)

    print("\n=== BERTScore post-processing complete ===")
    for r in summary:
        avg = r.get("avg_bertscore")
        avg_str = f"{avg:.4f}" if avg is not None else "N/A"
        print(f"  {Path(r['file']).name}: {r['updated']} rows updated, avg={avg_str}")


if __name__ == "__main__":
    main()
