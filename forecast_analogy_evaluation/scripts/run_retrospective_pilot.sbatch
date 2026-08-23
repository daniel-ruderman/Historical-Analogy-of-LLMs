#!/usr/bin/env bash
#SBATCH --job-name=forecast_pilot
#SBATCH --output=logs/forecast_pilot_%j.out
#SBATCH --error=logs/forecast_pilot_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu.q
#SBATCH --account=habib-lab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

# Usage (smoke):  sbatch scripts/run_retrospective_pilot.sbatch smoke
# Usage (pilot):  sbatch scripts/run_retrospective_pilot.sbatch pilot 20
# Usage (resume): sbatch scripts/run_retrospective_pilot.sbatch resume pilot_smoke_57457642 2

MODE="${1:-smoke}"
ARG2="${2:-2}"
ARG3="${3:-2}"

if [[ "$MODE" == "resume" ]]; then
    RUN_ID="$ARG2"
    LIMIT="$ARG3"
    EXTRA=(--resume --smoke --limit "$LIMIT")
else
    LIMIT="$ARG2"
    RUN_ID="pilot_${MODE}_${SLURM_JOB_ID}"
    EXTRA=()
    if [[ "$MODE" == "smoke" ]]; then
        EXTRA+=(--smoke --limit "$LIMIT")
    else
        EXTRA+=(--limit "$LIMIT")
    fi
fi

HAL_ROOT=/ems/elsc-labs/habib-n/yuval.rom/school/ai_agents/Historical-Analogy-of-LLMs
EVAL_ROOT="$HAL_ROOT/forecast_analogy_evaluation"
CONDA_SH=/ems/elsc-labs/habib-n/yuval.rom/miniconda3/etc/profile.d/conda.sh
ENV_PREFIX=/ems/elsc-labs/habib-n/yuval.rom/miniconda3/envs/historical-analogy-llms
OLLAMA_VERSION=0.32.15
OLLAMA_RUNTIME="$HAL_ROOT/models/ollama-linux-amd64-v${OLLAMA_VERSION}"
OLLAMA_BIN="$OLLAMA_RUNTIME/bin/ollama"

cd "$EVAL_ROOT"
mkdir -p logs

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID}"
echo "Date: $(date -Is)"
echo "Mode: $MODE  Run: $RUN_ID  Limit: $LIMIT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

source "$CONDA_SH"
conda activate "$ENV_PREFIX"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
pip install -q -r requirements.txt

export OLLAMA_MODELS="$HAL_ROOT/models"
export OLLAMA_HOST=127.0.0.1:11434
export LOCAL_BASE_URL="http://127.0.0.1:11434"

"$OLLAMA_BIN" serve >"logs/ollama_pilot_${SLURM_JOB_ID}.out" 2>"logs/ollama_pilot_${SLURM_JOB_ID}.err" &
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

python run_evaluation.py \
    --config configs/retrospective_pilot.yaml \
    --run-id "$RUN_ID" \
    --round-date 2024-07-21 \
    "${EXTRA[@]}"

echo "=== PILOT COMPLETE: $RUN_ID ==="
