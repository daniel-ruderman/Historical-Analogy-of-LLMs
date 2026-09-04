# Historical Analogy Forecast Evaluation

This subdirectory tests whether historical analogies improve LLM probabilistic
forecasting. It lives inside the main
[Historical-Analogy-of-LLMs](..) repository and shares the same git history,
Conda environment, and agentic pipeline.

## Research questions (priority order)

1. **Primary (practical):** Does the complete historical-analogy pipeline forecast
   better than a plain question-only LLM?
2. **Secondary (mechanistic):** When context, tools, compute, and output length are
   matched, does historical-analogy *content* beat non-analogy deliberation?
3. **Exploratory:** Where does any gain concentrate (domain, horizon, analogy quality)?

## Directory map

```
forecast_analogy_evaluation/
├── README.md                 ← you are here
├── run_evaluation.py         ← main entry point
├── docs/                     ← living protocol, assumptions, leakage, report
├── prompts/                  ← frozen, versioned treatment prompts
├── configs/                  ← run configurations
├── scripts/                  ← cluster sbatch templates
├── data/manifests/           ← provenance and inclusion ledger
├── analysis/                 ← scoring and aggregate analysis
├── runs/                     ← raw outputs (git-ignored)
└── results/                  ← derived tables and figures
```

## Quick start

From the repository root (or this directory):

```bash
# Dry-run (no GPU):
python forecast_analogy_evaluation/run_evaluation.py --dry-run --smoke --limit 2

# Retrospective pilot (requires Ollama + qwen3:8b):
cd forecast_analogy_evaluation
python run_evaluation.py --config configs/retrospective_pilot.yaml --limit 20

# Cluster (GPU):
sbatch scripts/run_retrospective_pilot.sbatch pilot 20
```

Read [`docs/README.md`](docs/README.md) for the full documentation index.

## Live ForecastBench submission

Generate the three ForecastBench upload files for a live round:

```bash
cd forecast_analogy_evaluation
python generate_forecastbench_submission.py \
  --config configs/prospective_forecastbench.yaml \
  --round-date 2026-08-30 \
  --organization "YOUR_ORG" \
  --model-organization "YOUR_ORG"
```

This writes raw artifacts under `runs/<run_id>/` and three upload files under
`runs/<run_id>/submissions/` named like:

- `2026-08-30.YOUR_ORG.1.json` → `plain`
- `2026-08-30.YOUR_ORG.2.json` → `matched_deliberation`
- `2026-08-30.YOUR_ORG.3.json` → `historical_analogy`

Useful flags:

- `--dry-run` for a schema rehearsal without Ollama/GPU
- `--limit N` for a small rehearsal instead of the full round
- `--include-reasoning` to include rationales in the upload files
- `--resume` to continue an interrupted live generation run

## Relationship to the analogy pipeline

Analogy generation uses the parent repo's
[`agentic_pipeline/`](../agentic_pipeline/). This evaluation:

- consumes analogy packets from that pipeline,
- runs the three-arm forecasting experiment (plain / deliberation / analogy),
- scores against ForecastBench / FutureEval resolutions,
- maintains its own provenance and report trail.

Pass@1 and MDS from the analogy-acquisition evaluation are **mechanism covariates
only**, not forecast endpoints.

## Status

| Component | Status |
|-----------|--------|
| Protocol & documentation | Ready |
| Runner & analysis code | Ready |
| Smoke test (2 questions) | Complete — `runs/pilot_smoke_57457642/` |
| 20-question pilot | Complete — `runs/pilot_pilot_57480457/` |
| Live ForecastBench submission generator | Ready |
| Live ForecastBench rounds | Not started |
