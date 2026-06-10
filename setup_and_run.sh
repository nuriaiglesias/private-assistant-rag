#!/usr/bin/env bash
# =============================================================================
# setup_and_run.sh — Full setup + evaluation for TFM RAG assistant
#
# Run this on the remote machine:
#   bash setup_and_run.sh
#
# You will be asked for:
#   - GitHub personal access token (to clone the private repo)
#   - Your .env file content (paste it when prompted)
#
# Everything else is automatic.
# =============================================================================

set -euo pipefail

REPO_URL="https://github.com/nuriaiglesias/private-assistant-rag.git"
REPO_DIR="$HOME/TFM"
LOG_FILE="$HOME/tfm_eval.log"

# ── colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

echo "============================================================"
echo "  TFM RAG Evaluation — Automated Setup"
echo "============================================================"
echo ""

# ── detect OS ─────────────────────────────────────────────────────────────────
if command -v apt-get &>/dev/null; then
    PKG_MANAGER="apt"
elif command -v dnf &>/dev/null; then
    PKG_MANAGER="dnf"
elif command -v yum &>/dev/null; then
    PKG_MANAGER="yum"
else
    error "Unsupported OS — only Debian/Ubuntu and RedHat/CentOS are supported."
fi
info "Detected package manager: $PKG_MANAGER"

# ── GPU detection ─────────────────────────────────────────────────────────────
HAS_GPU=false
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    HAS_GPU=true
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    info "GPU detected: $GPU_NAME — Ollama will use CUDA (much faster)"
else
    warn "No NVIDIA GPU detected — Ollama will use CPU (slower but works)"
fi

# ── install system dependencies ───────────────────────────────────────────────
info "Installing system dependencies..."
if [ "$PKG_MANAGER" = "apt" ]; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq curl git python3 python3-pip python3-venv \
        ca-certificates gnupg lsb-release jq 2>/dev/null
else
    sudo $PKG_MANAGER install -y curl git python3 python3-pip \
        ca-certificates gnupg jq 2>/dev/null
fi
success "System dependencies installed"

# ── install Docker ────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    info "Installing Docker..."
    curl -fsSL https://get.docker.com | sudo bash
    sudo usermod -aG docker "$USER"
    # Apply docker group without logout
    exec sg docker "$0 $*" 2>/dev/null || true
fi
# Start docker if not running
sudo systemctl start docker 2>/dev/null || sudo service docker start 2>/dev/null || true
success "Docker ready"

# ── install Ollama ────────────────────────────────────────────────────────────
if ! command -v ollama &>/dev/null; then
    info "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi
# Start Ollama in background if not running
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    info "Starting Ollama..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    sleep 5
fi
success "Ollama ready"

# ── pull Ollama model ─────────────────────────────────────────────────────────
info "Pulling model qwen2.5:0.5b-instruct (this may take a few minutes)..."
ollama pull qwen2.5:0.5b-instruct
success "Model ready"

# ── start Qdrant via Docker ───────────────────────────────────────────────────
info "Starting Qdrant vector database..."
if ! docker ps --format '{{.Names}}' | grep -q "^qdrant$"; then
    docker run -d \
        --name qdrant \
        --restart unless-stopped \
        -p 6333:6333 \
        -v "$HOME/qdrant_storage:/qdrant/storage" \
        qdrant/qdrant:latest \
        > /dev/null
    sleep 5
fi
# Wait for Qdrant to be healthy
for i in $(seq 1 30); do
    if curl -s http://localhost:6333/healthz | grep -q "pass"; then
        break
    fi
    sleep 2
done
success "Qdrant ready at http://localhost:6333"

# ── clone repository ──────────────────────────────────────────────────────────
if [ ! -d "$REPO_DIR/.git" ]; then
    echo ""
    echo "The repository is private. You need a GitHub personal access token."
    echo "Create one at: https://github.com/settings/tokens"
    echo "Required scope: repo (read access)"
    echo ""
    read -rsp "Paste your GitHub personal access token: " GH_TOKEN
    echo ""
    CLONE_URL="https://${GH_TOKEN}@github.com/nuriaiglesias/private-assistant-rag.git"
    info "Cloning repository..."
    git clone "$CLONE_URL" "$REPO_DIR"
    # Remove token from git config for security
    git -C "$REPO_DIR" remote set-url origin "$REPO_URL"
    success "Repository cloned to $REPO_DIR"
else
    info "Repository already exists, pulling latest..."
    git -C "$REPO_DIR" pull origin main
fi

# ── create .env file ──────────────────────────────────────────────────────────
if [ ! -f "$REPO_DIR/assistant/.env" ]; then
    echo ""
    echo "============================================================"
    echo "Paste the contents of the .env file below."
    echo "When done, press Enter then Ctrl+D on a new line."
    echo "============================================================"
    cat > "$REPO_DIR/assistant/.env"
    success ".env file created"
else
    warn ".env already exists, skipping"
fi

# Force timeout to 900s regardless of what's in .env
if ! grep -q "LLM_TIMEOUT_SECONDS" "$REPO_DIR/assistant/.env"; then
    echo "LLM_TIMEOUT_SECONDS=900" >> "$REPO_DIR/assistant/.env"
else
    sed -i 's/LLM_TIMEOUT_SECONDS=.*/LLM_TIMEOUT_SECONDS=900/' "$REPO_DIR/assistant/.env"
fi

# ── set up Python environment ─────────────────────────────────────────────────
cd "$REPO_DIR"
info "Creating Python virtual environment..."
python3 -m venv assistant/.venv
source assistant/.venv/bin/activate
pip install --upgrade pip -q
info "Installing Python dependencies (this takes a few minutes)..."
pip install -r assistant/requirements.txt -q
pip install -e assistant/src/ -q 2>/dev/null || true
success "Python environment ready"

# ── ingest documents into Qdrant ──────────────────────────────────────────────
info "Ingesting documents into Qdrant (downloads embedding model ~2.2GB)..."
python assistant/scripts/ingest/ingest_documents.py
success "Documents ingested into Qdrant"

# ── run evaluations ───────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete! Starting evaluation run."
echo "  Log file: $LOG_FILE"
echo "  Monitor: tail -f $LOG_FILE"
echo "  Safe to disconnect — it will keep running."
echo "============================================================"
echo ""

mkdir -p logs

nohup bash -c '
source assistant/.venv/bin/activate

log() { echo "[$(date "+%H:%M:%S")] $*"; }
run_step() {
    local label="$1"; shift
    log "START — $label"
    if "$@" 2>&1; then log "DONE  — $label"
    else log "ERROR — $label (exit $?). Continuing."; fi
    echo ""
}

log "============================================================"
log "Evaluation started"
log "============================================================"

run_step "B2-A: Dense semantic" \
    python assistant/scripts/eval/evaluate_answers.py --skip-bertscore

run_step "B2-B: Hybrid (dense + BM25)" \
    python assistant/scripts/eval/evaluate_answers.py --use-hybrid --skip-bertscore

run_step "B2-C: Hybrid + Reranking" \
    python assistant/scripts/eval/evaluate_answers.py --use-hybrid --use-reranking --skip-bertscore

run_step "B2-D: Hybrid + Reranking + ReAct Orchestrator" \
    python assistant/scripts/eval/evaluate_answers.py \
        --use-hybrid --use-reranking --use-orchestrator --limit 35 --skip-bertscore

run_step "B3: ReAct orchestrator evaluation" \
    python assistant/scripts/eval/evaluate_orchestrator.py \
        --use-hybrid --use-reranking --compare

run_step "XQuAD: Retrieval variants A/B/C" \
    python assistant/scripts/eval/evaluate_public_all_variants.py

run_step "XQuAD-D: Orchestrator answers" \
    python assistant/scripts/eval/evaluate_answers.py \
        --questions assistant/results/xquad_questions_for_answers.json \
        --use-hybrid --use-reranking --use-orchestrator --limit 25 --skip-bertscore

run_step "BERTScore: post-processing" \
    python assistant/scripts/eval/compute_bertscore.py

log "============================================================"
log "ALL DONE. Results in assistant/results/"
log "============================================================"
' > "$LOG_FILE" 2>&1 &

EVAL_PID=$!
echo "Evaluation running as PID $EVAL_PID"
echo "Monitor progress: tail -f $LOG_FILE"
echo ""
echo "You can safely close this SSH session."
echo "Results will be in: $REPO_DIR/assistant/results/"
