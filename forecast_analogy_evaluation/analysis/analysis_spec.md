# Analysis Specification

**Version:** 1.0  
**Last updated:** 2026-08-23  
**Implementation:** [`scoring.py`](scoring.py), [`aggregate.py`](aggregate.py)

---

## 1. Input data

### forecasts.jsonl (one row per question × condition)

Required fields:

| Field | Type | Description |
|-------|------|-------------|
| `question_id` | str | Stable benchmark ID |
| `condition` | str | `plain`, `matched_deliberation`, `historical_analogy`, ... |
| `p_yes` | float | Raw forecast (before clipping) |
| `p_yes_clipped` | float | Clipped to [0.01, 0.99] |
| `outcome` | int | 0 or 1 (after resolution) |
| `round_id` | str | ForecastBench round |
| `cluster_id` | str | Event/series cluster for dependence |
| `status` | str | `ok` or `failed` |

### resolutions.jsonl (one row per question)

| Field | Type | Description |
|-------|------|-------------|
| `question_id` | str | |
| `outcome` | int | 0 or 1 |
| `resolution_date` | str | ISO 8601 |

---

## 2. Per-question metrics

For each question `i` and condition `c`:

```
Brier(i, c) = (p_i,c − y_i)²

LogLoss(i, c) = −[y_i · log(p_i,c) + (1 − y_i) · log(1 − p_i,c)]
```

Use clipped probabilities for both metrics.

### Paired differences

```
Δ_practical(i) = Brier(i, analogy) − Brier(i, plain)
Δ_mechanism(i) = Brier(i, analogy) − Brier(i, matched_deliberation)
```

Negative values favor the analogy pipeline.

---

## 3. Aggregate estimation

### Primary estimand

```
Δ_practical = (1/N) Σ_i Δ_practical(i)
```

over all questions with valid forecasts in **both** arms of the contrast.

### Uncertainty

**Method:** Paired cluster-bootstrap

1. Resample clusters (round × event_series) with replacement, B = 10,000.
2. Within each bootstrap sample, include all questions from sampled clusters.
3. Compute `Δ` for each bootstrap replicate.
4. Report 95% CI as the 2.5th and 97.5th percentiles.

**Alternative:** Mixed-effects model with random intercept for cluster:

```
Brier(i,c) = β₀ + β₁·analogy_c + u_{cluster(i)} + ε_i
```

where `analogy_c` is a condition indicator. Report `β₁` and its SE.

Use bootstrap as primary; mixed model as sensitivity.

---

## 4. Hypothesis tests

| Contrast | H0 | Test | Adjustment |
|----------|----|------|------------|
| Primary (analogy vs plain) | Δ ≥ 0 | One-sided bootstrap p | None |
| Secondary (analogy vs delib) | Δ ≥ 0 | One-sided bootstrap p | Holm (within secondaries) |
| Log loss (analogy vs plain) | Δ ≥ 0 | One-sided bootstrap p | Holm |
| Calibration slope diff | = 0 | Two-sided | Exploratory |

One-sided p-value = fraction of bootstrap Δ ≥ 0 (primary) or ≤ 0 (if testing reverse).

---

## 5. Calibration analysis

Bin forecasts into deciles (0–10%, 10–20%, ..., 90–100%).

For each condition:

- **Calibration slope:** OLS of outcome on forecast across bins.
- **Calibration intercept:** should be ≈ 0 for well-calibrated forecasts.
- **ECE:** weighted mean |accuracy − confidence| across bins.

Report per condition; compare slopes descriptively.

---

## 6. Subgroup analyses (exploratory)

For each subgroup `g`:

```
Δ_practical(g) = mean_i∈g [ Brier(i, analogy) − Brier(i, plain) ]
```

with cluster-bootstrap CI within subgroup.

Prespecified subgroups (see protocol §8):

- `market`, `dataset`
- `horizon_short` (≤ 90d), `horizon_long` (> 365d)
- `analogy_relevant` (blind rating ≥ 3/5)
- `analogy_high_quality` (factuality + relevance ≥ 4/5)

**Do not** claim significance for subgroups without replication.

---

## 7. Coverage and missing data

Report per arm:

```
coverage(c) = |{i : status(i,c) = ok}| / N_eligible
```

**Primary analysis:** complete-case (questions with valid forecasts in both contrast arms).

**Sensitivity:** impute missing forecasts to p = 0.5 and recompute Δ.

---

## 8. Cost analysis

Per arm, report:

- Mean tokens_in, tokens_out
- Mean latency_ms
- Mean tool_calls (for deliberation/analogy arms)
- Total wall time

Cost-adjusted metric (exploratory):

```
Δ_cost_adjusted = Δ_practical / (cost_analogy / cost_plain)
```

---

## 9. Quality moderation (exploratory)

Merge quality audit ratings with per-question Δ:

```
Δ(i) = Brier(i, plain) − Brier(i, analogy)   [positive = analogy helped]
```

Regress `Δ(i)` on blind quality ratings (factuality, relevance, diversity).
Report R² and sign of coefficient.

---

## 10. Output files

| File | Content |
|------|---------|
| `results/paired_effects.csv` | Per-question losses and paired differences |
| `results/aggregate_summary.json` | Primary/secondary Δ, CIs, p-values |
| `results/subgroup_effects.csv` | Subgroup Δ with CIs |
| `results/calibration.csv` | Bin-level calibration data |
| `results/cost_summary.csv` | Token/latency aggregates |

---

## 11. Confirmatory vs exploratory labeling

Every result file must include a `"analysis_type"` field:

- `"confirmatory"` — primary and key secondary endpoints
- `"exploratory"` — subgroups, diagnostics, quality moderation, cost

Run analysis via:

```bash
python -m analysis.aggregate --run-id <run_id> --config configs/retrospective_pilot.yaml
```
