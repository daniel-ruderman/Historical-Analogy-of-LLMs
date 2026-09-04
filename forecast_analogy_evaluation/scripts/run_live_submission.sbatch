#!/usr/bin/env bash
#SBATCH --job-name=forecast_live
#SBATCH --output=logs/forecast_live_%j.out
#SBATCH --error=logs/forecast_live_%j.err
#SBATCH --time=6-00:00:00
#SBATCH --partition=gpu.q
#SBATCH --account=habib-lab
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

# Usage:
#   sbatch scripts/run_live_submission.sbatch <round_date> <organization> [mode]
#   sbatch scripts/run_live_submission.sbatch 2026-08-30 "HebU-AnalogAI" market
#   sbatch scripts/run_live_submission.sbatch 2026-08-30 "HebU-AnalogAI" dataset_sample
#
# Resume an interrupted run:
#   sbatch scripts/run_live_submission.sbatch 2026-08-30 "HebU-AnalogAI" market resume <run_id>
#
# Optional Slurm overrides at submit time (examples):
#   sbatch --exclude=elscn-108 scripts/run_live_submission.sbatch ...
#   sbatch --nodelist=elscn-109 scripts/run_live_submission.sbatch ...

ROUND_DATE="${1:?Usage: sbatch run_live_submission.sbatch <round_date> <organization> [mode] [resume <run_id>]}"
ORGANIZATION="${2:?Missing organization argument}"
MODE="${3:-all}"
RESUME_MODE="${4:-}"
RESUME_RUN_ID="${5:-}"

if [[ "$MODE" != "all" && "$MODE" != "market" && "$MODE" != "dataset_sample" ]]; then
    echo "Invalid mode '$MODE'. Use one of: all, market, dataset_sample"
    exit 2
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

# Unique local port per job so two jobs on the same node never share Ollama.
OLLAMA_PORT=$((20000 + (SLURM_JOB_ID % 20000)))
OLLAMA_HOST_ADDR="127.0.0.1:${OLLAMA_PORT}"

echo "Host: $(hostname)"
echo "Job:  ${SLURM_JOB_ID}"
echo "Date: $(date -Is)"
echo "Round: $ROUND_DATE  Org: $ORGANIZATION  Mode: $MODE"
echo "Ollama: ${OLLAMA_HOST_ADDR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
if [[ -n "$RESUME_RUN_ID" ]]; then
    echo "Resuming run: $RESUME_RUN_ID"
fi
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

"$OLLAMA_BIN" serve >"logs/ollama_live_${SLURM_JOB_ID}.out" 2>"logs/ollama_live_${SLURM_JOB_ID}.err" &
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

# Build submission generator args
SUBMIT_ARGS=(
    --config   configs/prospective_forecastbench.yaml
    --round-date "$ROUND_DATE"
    --organization "$ORGANIZATION"
    --model-organization "$ORGANIZATION"
    --include-reasoning
)

if [[ "$MODE" == "market" ]]; then
    SUBMIT_ARGS+=(--source-categories market)
fi

if [[ "$MODE" == "dataset_sample" ]]; then
    # Method-evaluation subset: sample dataset base questions and first horizons.
    SUBMIT_ARGS+=(--source-categories dataset --dataset-question-limit 80 --dataset-horizons-per-question 2 --seed 42)
fi

RUN_ID="live_${MODE}_${ROUND_DATE//-/}_${SLURM_JOB_ID}"
SUBMIT_ARGS+=(--run-id "$RUN_ID")

if [[ "$RESUME_MODE" == "resume" && -n "$RESUME_RUN_ID" ]]; then
    SUBMIT_ARGS+=(--run-id "$RESUME_RUN_ID" --resume)
    RUN_ID="$RESUME_RUN_ID"
fi

python generate_forecastbench_submission.py "${SUBMIT_ARGS[@]}"
python -m analysis.live_summary_plots --run-id "$RUN_ID" --output-name "live_summary_plots_${MODE}.png"

echo "=== LIVE SUBMISSION COMPLETE: $ROUND_DATE ==="
echo "Run ID: $RUN_ID"
echo "Artifacts: runs/${RUN_ID}/"
echo "Plots:     results/${RUN_ID}/live_summary_plots_${MODE}.png"
