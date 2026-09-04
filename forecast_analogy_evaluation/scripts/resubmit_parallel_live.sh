#!/usr/bin/env bash
set -euo pipefail
EVAL=/ems/elsc-labs/habib-n/yuval.rom/school/ai_agents/Historical-Analogy-of-LLMs/forecast_analogy_evaluation
cd "$EVAL"
sed -i 's/\r$//' scripts/run_live_submission.sbatch
J1=$(sbatch --parsable scripts/run_live_submission.sbatch 2026-08-16 yuval-rom market resume live_market_20260816_57579130)
echo "MARKET_JOB=$J1"
# Wait until market has a node (or timeout)
NODE=""
for i in $(seq 1 60); do
  NODE=$(squeue -j "$J1" -h -o '%N' | tr -d ' ')
  ST=$(squeue -j "$J1" -h -o '%t' | tr -d ' ')
  echo "wait $i: job=$J1 state=$ST node=$NODE"
  if [[ -n "$NODE" && "$NODE" != "(null)" ]]; then
    break
  fi
  sleep 5
done
if [[ -z "$NODE" || "$NODE" == "(null)" ]]; then
  # Fall back: keep dataset off the historically contended node
  EXCLUDE=elscn-108
else
  EXCLUDE="$NODE"
fi
J2=$(sbatch --parsable --exclude="$EXCLUDE" scripts/run_live_submission.sbatch 2026-08-16 yuval-rom dataset_sample resume live_dataset_sample_20260816_57579131)
echo "DATASET_JOB=$J2 exclude=$EXCLUDE"
squeue -j "$J1,$J2"
