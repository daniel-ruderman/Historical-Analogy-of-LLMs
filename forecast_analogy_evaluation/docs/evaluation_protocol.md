# Evaluation Protocol

**Version:** 1.0  
**Last updated:** 2026-08-23  
**Status:** Frozen for retrospective pilot; prospective preregistration in [`prospective_preregistration.md`](prospective_preregistration.md)

---

## 1. Hypotheses (priority order)

### H1 — Primary practical hypothesis

The complete historical-analogy pipeline produces **lower Brier loss** than a plain
question-only LLM on the same forecasting questions.

- **Estimand:** `Δ_practical = mean_i [ Brier(analogy_pipeline, q_i) − Brier(plain, q_i) ]`
- **Direction:** Negative Δ favors the analogy pipeline.
- **Interpretation:** Answers “Is this system useful in practice?”

### H2 — Secondary mechanistic hypothesis

When both systems receive matched context, tools, compute, and output length,
historical-analogy **content** produces lower Brier loss than non-analogy deliberation.

- **Estimand:** `Δ_mechanism = mean_i [ Brier(analogy_pipeline, q_i) − Brier(matched_deliberation, q_i) ]`
- **Direction:** Negative Δ means analogies add value beyond generic reasoning.
- **Interpretation:** Answers “Did analogies cause the gain, or just extra deliberation?”

### H3 — Exploratory (not confirmatory)

Gains concentrate in questions where analogies are structurally relevant, factually
grounded, and of high quality. See §8 for prespecified subgroups.

---

## 2. Treatment arms

Every eligible question is forecast under all arms in **fresh contexts** with
**randomized arm order** (within each question batch).

| Arm ID | Name | What the forecaster receives |
|--------|------|------------------------------|
| `plain` | Plain LLM | Question text, background, resolution criteria, forecast timestamp |
| `matched_deliberation` | Matched deliberation | Same as plain + a deliberation packet (base rates, causal drivers, counterarguments) with **no historical analogies**; matched tool budget, calls, and output tokens |
| `historical_analogy` | Analogy pipeline | Same as plain + a rich analogy packet from the agentic pipeline (winner, similarities, differences, anti-analogies, limitations) |

### Diagnostic arms (retrospective pilot only)

| Arm ID | Purpose |
|--------|---------|
| `shuffled_analogy` | Same token budget as analogy arm but analogy packet drawn from a different question in the same domain — negative control |
| `analogy_name_only` | Only the winning analogy event name, no structured comparison — tests whether richness matters |

Diagnostic arms are **not** part of the confirmatory prospective study unless explicitly added in a protocol amendment.

---

## 3. Two-stage architecture

```
Forecast question
       │
       ├─ plain ──────────────────────────────► Forecaster ──► p(YES)
       │
       ├─ matched_deliberation
       │       └─ Deliberation generator ──► packet ──► Forecaster ──► p(YES)
       │
       └─ historical_analogy
               └─ Agentic pipeline ──► analogy packet ──► Forecaster ──► p(YES)
```

**Separation rule:** Analogy/deliberation generation and final forecasting use the
**same LLM model** but **separate prompts and contexts**. The generated packet is
saved before forecasting so it can be audited, reused, and scored for quality.

Prompts are frozen in [`../prompts/`](../prompts/) with version tags in filenames.

---

## 4. Question eligibility

### 4.1 General requirements

A question is **eligible** if all of the following hold:

1. **Binary outcome** (for the primary analysis; multi-outcome reserved for FutureEval replication).
2. **Resolution known** (retrospective) or **not yet known at forecast time** (prospective).
3. **Forecast timestamp** `t_forecast` is recorded and precedes resolution.
4. **Background and resolution criteria** are available in the benchmark snapshot.
5. The question was **published after the evaluated model's knowledge cutoff** (retrospective) or is **currently unresolved** (prospective).

### 4.2 Retrospective-specific (interim pilot)

- Source: archived [ForecastBench](https://www.forecastbench.org/) question + resolution sets.
- Include **all** eligible binary questions in the snapshot — no cherry-picking.
- Reconstruct information available at `t_forecast`; see [`data_leakage.md`](data_leakage.md).
- Exclude questions where the outcome was widely known before `t_forecast` despite formal resolution later.

### 4.3 Prospective-specific (confirmatory)

- Source: biweekly ForecastBench rounds (500 questions per round).
- Forecast before resolution; timestamp and freeze all artifacts.
- Submit up to three arms per round (ForecastBench team limit).
- Analyze raw paired per-question Brier losses locally — do not rely on leaderboard rank or Brier Index for inference.

### 4.4 Exclusion ledger

Every inclusion/exclusion is recorded in
[`../data/manifests/inclusion_exclusion_ledger.csv`](../data/manifests/inclusion_exclusion_ledger.csv)
**before** examining condition outcomes. Valid exclusion reasons:

- `post_cutoff_violation` — question or outcome leaked before forecast time
- `missing_resolution` — outcome unavailable
- `non_binary` — not a binary question (primary analysis)
- `parse_failure_all_arms` — model failed on all arms (report separately)
- `contamination_detected` — see leakage protocol
- `insufficient_context` — missing background or resolution criteria

---

## 5. Forecast output schema

Every forecast must produce:

```json
{
  "question_id": "fb:2026-03-15:market:12345",
  "condition": "historical_analogy",
  "p_yes": 0.37,
  "rationale": "One or two sentences.",
  "forecast_timestamp": "2026-03-15T12:00:00Z",
  "prompt_version": "forecast_v1",
  "model": "qwen3:8b",
  "tokens_in": 1842,
  "tokens_out": 87,
  "latency_ms": 4200
}
```

Constraints:

- `p_yes` ∈ [0.01, 0.99] after clipping (record both raw and clipped).
- Refusals and parse failures are recorded with `status: "failed"` — never silently dropped.
- The **forecasting prompt is identical** across conditions except for the prepended analysis packet.

---

## 6. Endpoints

### 6.1 Primary endpoint (confirmatory)

**Paired mean Brier difference — practical:**

```
Δ_practical = (1/N) Σ_i [ (p_i,analogy − y_i)² − (p_i,plain − y_i)² ]
```

where `y_i ∈ {0, 1}` is the resolved outcome and `p_i,*` is the clipped forecast.

### 6.2 Key secondary endpoint

**Paired mean Brier difference — mechanistic:**

```
Δ_mechanism = (1/N) Σ_i [ (p_i,analogy − y_i)² − (p_i,delib − y_i)² ]
```

### 6.3 Other secondary endpoints

| Metric | Definition | Notes |
|--------|------------|-------|
| Log loss | `−[y·log(p) + (1−y)·log(1−p)]` | Clipped at [0.01, 0.99] |
| Calibration slope | OLS of outcome on forecast across bins | Per condition |
| Coverage | Fraction of questions with valid forecasts | Must be ≥ 95% per arm |
| Cost | tokens × price + latency | For cost-adjusted comparison |
| ECE | Expected calibration error | Exploratory |

### 6.4 What we do NOT use as inferential endpoints

- ForecastBench difficulty-adjusted Brier (descriptive only)
- Brier Index (descriptive only — nonlinear transform of Brier)
- Leaderboard rank
- Pass@1 or MDS from the analogy repo (mechanism covariates only)

---

## 7. Analysis plan

See [`../analysis/analysis_spec.md`](../analysis/analysis_spec.md) for full detail.

Summary:

1. Compute per-question paired losses for each contrast.
2. Estimate `Δ` with question-level paired bootstrap (10,000 resamples) or mixed model.
3. **Cluster** by ForecastBench round and underlying event/time-series ID.
4. Report 95% CI and two-sided p-value for primary and key secondary endpoints.
5. Multiple-comparison adjustment: primary endpoint is unadjusted; secondary endpoints use Holm correction across {mechanistic, log loss, calibration}.
6. Prespecified subgroups (§8) are reported but not used to claim significance without replication.

### Power analysis

Use retrospective pilot paired-loss variance (`σ²_Δ`) to set live-round count:

```
N_needed ≈ (z_{α/2} + z_β)² × σ²_Δ / δ²
```

where `δ` is the minimum meaningful effect (default: 0.01 Brier points). Record estimate in
[`../analysis/power_analysis_template.md`](../analysis/power_analysis_template.md).

### Stopping rule (prospective)

Fix the number of ForecastBench rounds **before** examining outcomes. Default: **6 rounds**
(~3,000 binary questions if all resolve). No optional stopping on positive results.

---

## 8. Prespecified subgroups

| Label | Definition | Rationale |
|-------|------------|-----------|
| `market` | ForecastBench market-sourced questions | Crowd-informed, harder to beat |
| `dataset` | ForecastBench templated time-series questions | More base-rate-friendly |
| `horizon_short` | Resolution ≤ 90 days | Near-term |
| `horizon_long` | Resolution > 365 days | Structural uncertainty |
| `analogy_relevant` | Blind rating ≥ 3/5 on structural relevance | Tests H3 |
| `analogy_high_quality` | Factuality + relevance both ≥ 4/5 | Quality moderation |

Subgroup labels for `analogy_relevant` and `analogy_high_quality` are assigned **before**
forecast outcomes are examined (blind audit on analogy packets).

---

## 9. Mechanism and quality audit

For every `historical_analogy` forecast, save the full packet:

- winning analogy name and dimensions
- similarities, differences, counterexamples, limitations
- evidence URLs and verification status
- ReAct tool steps (query, tool, results)

A random sample (minimum 50 packets per stage) is blindly rated on:

1. Factuality (1–5)
2. Temporal validity — analogy precedes forecast time (yes/no)
3. Structural relevance to the forecast question (1–5)
4. Diversity — not a surface-level label match (1–5)
5. Forecaster usage — did the rationale reference the analogy? (yes/no)

Relate quality scores to per-question `Δ_practical`.

---

## 10. Evaluation stages

| Stage | Config | Doc |
|-------|--------|-----|
| Retrospective pilot | [`../configs/retrospective_pilot.yaml`](../configs/retrospective_pilot.yaml) | [`retrospective_pilot.md`](retrospective_pilot.md) |
| Prospective confirmatory | [`../configs/prospective_forecastbench.yaml`](../configs/prospective_forecastbench.yaml) | [`prospective_preregistration.md`](prospective_preregistration.md) |
| External replication | [`../configs/futureeval_replication.yaml`](../configs/futureeval_replication.yaml) | [`futureeval_replication.md`](futureeval_replication.md) |

---

## 11. Model and compute defaults

Initial evaluation uses the parent repo defaults:

| Setting | Value | Source |
|---------|-------|--------|
| LLM | `qwen3:8b` via Ollama | `hal/project_defaults.py` |
| Temperature (forecast) | 0.1 | Same as generation |
| Max output tokens | 4096 | Same as generation |
| Search | Wikipedia (historical only) | Leakage policy |
| ReAct max steps | 4 | Same as agentic pipeline |

Any change to model, temperature, or token budget requires a new prompt/config version and a
log entry in [`decisions.md`](decisions.md).

---

## 12. Confirmatory vs exploratory

| Analysis | Type |
|----------|------|
| Δ_practical (analogy vs plain) | Confirmatory |
| Δ_mechanism (analogy vs matched deliberation) | Confirmatory (secondary) |
| Diagnostic arms (shuffled, name-only) | Exploratory (pilot only) |
| Subgroups | Exploratory (reported, not headline) |
| Analogy quality moderation | Exploratory |
| Cost-adjusted performance | Exploratory |

Any deviation from this table must be logged before running the analysis.
