#!/usr/bin/env bash
#SBATCH --job-name=hal_smoke
#SBATCH --output=logs/hal_smoke_%j.out
#SBATCH --error=logs/hal_smoke_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu.q
#SBATCH --account=habib-lab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

ROOT=/ems/elsc-labs/habib-n/yuval.rom/school/ai_agents/Historical-Analogy-of-LLMs
CONDA_SH=/ems/elsc-labs/habib-n/yuval.rom/miniconda3/etc/profile.d/conda.sh
ENV_PREFIX=/ems/elsc-labs/habib-n/yuval.rom/miniconda3/envs/historical-analogy-llms
OLLAMA_VERSION=0.32.15
OLLAMA_RUNTIME="$ROOT/models/ollama-linux-amd64-v${OLLAMA_VERSION}"
OLLAMA_BIN="$OLLAMA_RUNTIME/bin/ollama"
DOWNLOAD_DIR="$ROOT/models/downloads"
ARCHIVE="$DOWNLOAD_DIR/ollama-linux-amd64.tar.zst"
CHECKSUMS="$DOWNLOAD_DIR/sha256sum.txt"

cd "$ROOT"
mkdir -p logs "$DOWNLOAD_DIR"

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
echo "Date: $(date -Is)"
echo "Project: $ROOT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# The Windows Conda environment is not portable to Linux. Create the shared
# Linux environment once, from a compute node, and reuse it on later jobs.
source "$CONDA_SH"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    conda create -y -p "$ENV_PREFIX" python=3.11 pip
fi
conda activate "$ENV_PREFIX"
export PYTHONNOUSERSITE=1
python -m pip install -r requirements_project.txt
echo "Python: $(command -v python)"
python --version

# Install a pinned Linux Ollama runtime under the already git-ignored models/
# directory. Verify the official release checksum before extraction.
if [[ ! -x "$OLLAMA_BIN" ]]; then
    curl --fail --location --retry 3 \
        "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" \
        --output "${ARCHIVE}.tmp"
    mv "${ARCHIVE}.tmp" "$ARCHIVE"
    curl --fail --location --retry 3 \
        "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/sha256sum.txt" \
        --output "$CHECKSUMS"
    (
        cd "$DOWNLOAD_DIR"
        sha256sum --ignore-missing --check "$(basename "$CHECKSUMS")"
    )
    mkdir -p "$OLLAMA_RUNTIME"
    zstd --decompress --stdout "$ARCHIVE" | tar -xf - -C "$OLLAMA_RUNTIME"
fi

"$OLLAMA_BIN" --version
export OLLAMA_MODELS="$ROOT/models"
export OLLAMA_HOST=127.0.0.1:11434
"$OLLAMA_BIN" serve >"logs/ollama_${SLURM_JOB_ID}.out" 2>"logs/ollama_${SLURM_JOB_ID}.err" &
OLLAMA_PID=$!
cleanup() {
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
    if curl --fail --silent http://127.0.0.1:11434/api/tags >/dev/null; then
        break
    fi
    sleep 2
done
curl --fail --silent http://127.0.0.1:11434/api/tags >/dev/null
"$OLLAMA_BIN" show qwen3:8b >/dev/null

echo "=== OFFLINE DRY-RUN SMOKE TEST ==="
python examples/run_all_methods.py \
    --methods agentic \
    --dataset popular \
    --index 0 \
    --smoke \
    --dry-run \
    --output "logs/smoke_dry_${SLURM_JOB_ID}.jsonl"

echo "=== REAL QWEN3:8B SMOKE TEST ==="
python examples/run_all_methods.py \
    --methods agentic \
    --dataset popular \
    --index 0 \
    --smoke \
    --output "logs/smoke_qwen3_8b_${SLURM_JOB_ID}.jsonl"

echo "=== BOTH SMOKE TESTS PASSED ==="
