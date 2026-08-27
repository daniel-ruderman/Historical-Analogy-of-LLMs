# Results

Derived tables and figures. Every file here must link back to:

- the run manifest in `runs/<run_id>/manifest.json`, and
- the analysis specification in `analysis/`.

## Expected outputs

| File | Description |
|------|-------------|
| `paired_effects.csv` | Per-question Brier losses and paired differences by condition |
| `aggregate_summary.json` | Primary and secondary endpoint estimates with CIs |
| `subgroup_effects.csv` | Prespecified subgroup analyses |
| `calibration.csv` | Calibration slope/intercept by condition |
| `cost_summary.csv` | Tokens, latency, tool calls by condition |
