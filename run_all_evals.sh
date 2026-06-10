#!/usr/bin/env bash
# run_all_evals.sh — runs all evaluation blocks sequentially.
# Launch with:  nohup bash run_all_evals.sh > logs/eval_run.log 2>&1 &
# Check progress: tail -f logs/eval_run.log

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$REPO_ROOT/assistant/.venv/bin/activate"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

source "$VENV"

# ── helpers ───────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_step() {
    local label="$1"; shift
    log "START — $label"
    if "$@" 2>&1; then
        log "DONE  — $label"
    else
        log "ERROR — $label (exit $?). Continuing to next step."
    fi
    echo ""
}

check_service() {
    local name="$1" url="$2"
    if curl -s --max-time 5 "$url" | grep -q "."; then
        log "$name is running ✓"
    else
        log "WARNING: $name not reachable at $url — results may fail."
    fi
}

# ── pre-flight checks ─────────────────────────────────────────────────────────

log "============================================================"
log "TFM evaluation run started"
log "============================================================"
check_service "Qdrant" "http://localhost:6333/healthz"
check_service "Ollama" "http://localhost:11434/api/tags"
echo ""

# ── BLOCK 2 — Answer quality (A, B, C, D) ────────────────────────────────────
# --skip-bertscore avoids loading the ~440MB BERTScore model during answer gen.
# BERTScore is computed separately in the post-processing step below.

log "=== BLOCK 2: Answer quality (calls LLM) ==="

run_step "B2-A: Dense semantic" \
    python assistant/scripts/eval/evaluate_answers.py \
        --skip-bertscore

run_step "B2-B: Hybrid (dense + BM25)" \
    python assistant/scripts/eval/evaluate_answers.py \
        --use-hybrid --skip-bertscore

run_step "B2-C: Hybrid + Reranking" \
    python assistant/scripts/eval/evaluate_answers.py \
        --use-hybrid --use-reranking --skip-bertscore

run_step "B2-D: Hybrid + Reranking + ReAct Orchestrator" \
    python assistant/scripts/eval/evaluate_answers.py \
        --use-hybrid --use-reranking --use-orchestrator --limit 35 --skip-bertscore

# ── BLOCK 3 — Orchestrator functional evaluation ──────────────────────────────

log "=== BLOCK 3: Orchestrator evaluation ==="

run_step "B3: ReAct orchestrator (hybrid + reranking)" \
    python assistant/scripts/eval/evaluate_orchestrator.py \
        --use-hybrid --use-reranking --compare

# ── PUBLIC DATASET — XQuAD-es (A, B, C) ──────────────────────────────────────

log "=== PUBLIC DATASET: XQuAD-es retrieval (A, B, C) ==="

run_step "XQuAD: Variants A/B/C" \
    python assistant/scripts/eval/evaluate_public_all_variants.py

# ── PUBLIC DATASET — XQuAD-es Variant D (answer quality) ─────────────────────

log "=== PUBLIC DATASET: XQuAD-es answers — Variant D ==="

run_step "XQuAD-D: Hybrid + Reranking + Orchestrator" \
    python assistant/scripts/eval/evaluate_answers.py \
        --questions assistant/results/xquad_questions_for_answers.json \
        --use-hybrid --use-reranking --use-orchestrator --limit 25 --skip-bertscore

# ── BLOCK 4 — BERTScore post-processing (no BGE-M3 in memory) ────────────────
# Compute BERTScore from saved JSONL files only (no embedding model needed).

log "=== BLOCK 4: BERTScore post-processing ==="

run_step "BERTScore: compute from saved JSONL results" \
    python assistant/scripts/eval/compute_bertscore.py

# ── done ─────────────────────────────────────────────────────────────────────

log "============================================================"
log "All evaluation steps completed. Results in assistant/results/"
log "============================================================"
