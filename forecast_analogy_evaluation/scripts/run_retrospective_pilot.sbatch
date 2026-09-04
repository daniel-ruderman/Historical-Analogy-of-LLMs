#!/usr/bin/env bash
#SBATCH --job-name=forecast_retro
#SBATCH --output=logs/forecast_retro_%j.out
#SBATCH --error=logs/forecast_retro_%j.err
#SBATCH --time=6-00:00:00
#SBATCH --partition=gpu.q
#SBATCH --account=habib-lab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

# Usage:
#   sbatch scripts/run_retrospective_pilot.sbatch <mode> [round_date]
#   sbatch scripts/run_retrospective_pilot.sbatch market 2026-08-02
#   sbatch scripts/run_retrospective_pilot.sbatch dataset_sample 2026-08-02
#   sbatch scripts/run_retrospective_pilot.sbatch pilot 20
#   sbatch scripts/run_retrospective_pilot.sbatch smoke 2
#   sbatch scripts/run_retrospective_pilot.sbatch resume <run_id> [limit]

MODE="${1:-smoke}"
ARG2="${2:-}"
ARG3="${3:-}"

HAL_ROOT=/ems/elsc-labs/habib-n/yuval.rom/school/ai_agents/Historical-Analogy-of-LLMs
EVAL_ROOT="$HAL_ROOT/forecast_analogy_evaluation"
CONDA_SH=/ems/elsc-labs/habib-n/yuval.rom/miniconda3/etc/profile.d/conda.sh
ENV_PREFIX=/ems/elsc-labs/habib-n/yuval.rom/miniconda3/envs/historical-analogy-llms
OLLAMA_VERSION=0.32.15
OLLAMA_RUNTIME="$HAL_ROOT/models/ollama-linux-amd64-v${OLLAMA_VERSION}"
OLLAMA_BIN="$OLLAMA_RUNTIME/bin/ollama"

cd "$EVAL_ROOT"
mkdir -p logs

# Unique local port per job so two jobs on the same node never share Ollama.
OLLAMA_PORT=$((20000 + (SLURM_JOB_ID % 20000)))
OLLAMA_HOST_ADDR="127.0.0.1:${OLLAMA_PORT}"

ROUND_DATE="2026-08-02"
EXTRA=()
RUN_ID=""

case "$MODE" in
    resume)
        RUN_ID="${ARG2:?Missing run_id for resume}"
        LIMIT="${ARG3:-}"
        EXTRA+=(--resume)
        if [[ -n "$LIMIT" ]]; then
            EXTRA+=(--limit "$LIMIT")
        fi
        ;;
    smoke)
        LIMIT="${ARG2:-2}"
        RUN_ID="pilot_smoke_${SLURM_JOB_ID}"
        EXTRA+=(--smoke --limit "$LIMIT")
        ;;
    pilot)
        LIMIT="${ARG2:-20}"
        RUN_ID="pilot_pilot_${SLURM_JOB_ID}"
        EXTRA+=(--limit "$LIMIT")
        ;;
    market)
        ROUND_DATE="${ARG2:-2026-08-02}"
        RUN_ID="retro_market_${ROUND_DATE//-/}_${SLURM_JOB_ID}"
        EXTRA+=(--source-categories market)
        ;;
    dataset_sample)
        ROUND_DATE="${ARG2:-2026-08-02}"
        RUN_ID="retro_dataset_sample_${ROUND_DATE//-/}_${SLURM_JOB_ID}"
        EXTRA+=(--source-categories dataset --dataset-question-limit 80 --seed 42)
        ;;
    *)
        echo "Invalid mode '$MODE'. Use: smoke|pilot|market|dataset_sample|resume"
        exit 2
        ;;
esac

echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID}"
echo "Date: $(date -Is)"
echo "Mode: $MODE  Round: $ROUND_DATE  Run: $RUN_ID"
echo "Ollama: ${OLLAMA_HOST_ADDR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

source "$CONDA_SH"
conda activate "$ENV_PREFIX"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
pip install -q -r requirements.txt
pip install -q matplotlib

export OLLAMA_MODELS="$HAL_ROOT/models"
export OLLAMA_HOST="$OLLAMA_HOST_ADDR"
export LOCAL_BASE_URL="http://${OLLAMA_HOST_ADDR}"

"$OLLAMA_BIN" serve >"logs/ollama_retro_${SLURM_JOB_ID}.out" 2>"logs/ollama_retro_${SLURM_JOB_ID}.err" &
OLLAMA_PID=$!
cleanup() {
    kill "$OLLAMA_PID" 2>/dev/null || true
    wait "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
    if curl --fail --silent "http://${OLLAMA_HOST_ADDR}/api/tags" >/dev/null; then
        break
    fi
    sleep 2
done
curl --fail --silent "http://${OLLAMA_HOST_ADDR}/api/tags" >/dev/null
"$OLLAMA_BIN" show qwen3:8b >/dev/null
echo "Ollama ready on ${OLLAMA_HOST_ADDR}"

python run_evaluation.py \
    --config configs/retrospective_pilot.yaml \
    --run-id "$RUN_ID" \
    --round-date "$ROUND_DATE" \
    "${EXTRA[@]}"

python -m analysis.resolved_summary_plots \
    --run-id "$RUN_ID" \
    --output-name "resolved_summary_plots_${MODE}.png"

echo "=== RETROSPECTIVE COMPLETE: $RUN_ID ==="
echo "Artifacts: runs/${RUN_ID}/"
echo "Results:   results/${RUN_ID}/"
echo "Plots:     results/${RUN_ID}/resolved_summary_plots_${MODE}.png"
