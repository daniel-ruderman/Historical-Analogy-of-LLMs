# Prospective Preregistration (Draft)

**Version:** 1.0-draft  
**Last updated:** 2026-08-23  
**Status:** Draft — finalize after retrospective pilot completes  
**Config:** [`../configs/prospective_forecastbench.yaml`](../configs/prospective_forecastbench.yaml)

> This document will be frozen and dated before the first scored live round.
> Until then, it is a draft subject to revision based on pilot findings.

---

## Study title

Do Historical Analogies Improve LLM Forecasting? A Preregistered Three-Arm Study on ForecastBench

---

## Authors

[To be filled]

---

## Hypotheses

### Primary (practical)

The complete historical-analogy pipeline achieves lower mean Brier loss than a plain
question-only LLM on ForecastBench binary questions.

- H0: `Δ_practical ≥ 0`
- H1: `Δ_practical < 0`

### Secondary (mechanistic)

When context, tools, compute, and output length are matched, historical-analogy content
achieves lower mean Brier loss than non-analogy deliberation.

- H0: `Δ_mechanism ≥ 0`
- H1: `Δ_mechanism < 0`

---

## Design

- **Benchmark:** [ForecastBench](https://www.forecastbench.org/) biweekly rounds
- **Unit of analysis:** binary forecast question
- **Design:** within-question, three-arm paired comparison
- **Arms:** `plain`, `matched_deliberation`, `historical_analogy`
- **Randomization:** arm order randomized within each question
- **Blinding:** forecast scoring is automated (Brier loss); analogy quality audit is blinded

---

## Sample

- **Rounds:** 6 consecutive biweekly ForecastBench rounds (~3,000 binary questions)
- **Stopping rule:** fixed at 6 rounds; no optional stopping
- **Inclusion:** all binary questions in each round that meet eligibility criteria (§4 of
  [`evaluation_protocol.md`](evaluation_protocol.md))
- **Power:** computed from retrospective pilot variance; target 80% power to detect
  |Δ| = 0.01 Brier points at α = 0.05

---

## Model and prompts

| Component | Setting |
|-----------|---------|
| Model | qwen3:8b via Ollama (pinned version recorded in manifest) |
| Temperature | 0.1 |
| Analogy pipeline | agentic (2 refinement rounds, ReAct max 4 steps) |
| Prompts | frozen v1 in `prompts/` |
| Search | Wikipedia, historical sources only (leakage policy enforced) |

---

## Primary endpoint

Paired mean Brier difference:

```
Δ_practical = mean_i [ Brier(analogy_i) − Brier(plain_i) ]
```

Clipped probabilities at [0.01, 0.99]. Negative values favor the analogy pipeline.

---

## Secondary endpoints

1. `Δ_mechanism = mean_i [ Brier(analogy_i) − Brier(matched_deliberation_i) ]`
2. Paired mean log loss difference (analogy vs plain)
3. Coverage rate per arm
4. Cost (tokens + wall time) per arm

---

## Analysis plan

1. Compute per-question Brier losses for each arm.
2. Estimate `Δ_practical` with cluster-bootstrap (cluster = ForecastBench round × event/series).
3. Report 95% CI and one-sided p-value for primary hypothesis.
4. Estimate `Δ_mechanism` with Holm-adjusted p-value.
5. Prespecified subgroups (exploratory): market vs dataset, horizon short vs long.
6. No imputation for missing forecasts — report coverage and analyze complete cases;
   sensitivity analysis with 0.5 imputation for missing values.

---

## Submission plan

ForecastBench allows **3 submissions per team per round**:

| Submission slot | Arm |
|----------------|-----|
| 1 | `plain` |
| 2 | `matched_deliberation` |
| 3 | `historical_analogy` |

Register team via ForecastBench submission process:
[How to submit](https://github.com/forecastingresearch/forecastbench/wiki/How-to-submit-to-ForecastBench)

Also score locally using resolution data from
[forecastbench-datasets](https://github.com/forecastingresearch/forecastbench-datasets).

---

## Deviations

Any deviation from this preregistration must be:

1. Logged in [`decisions.md`](decisions.md) **before** the affected analysis is run.
2. Labeled "confirmatory" or "exploratory" in the final report.
3. Reported transparently with rationale.

---

## Timeline

| Milestone | Target |
|-----------|--------|
| Retrospective pilot complete | T + 2 weeks |
| Preregistration frozen | T + 3 weeks |
| First live round submitted | T + 4 weeks |
| 6 rounds complete | T + 16 weeks |
| Sufficient resolutions for scoring | T + 20–30 weeks |
| Final report | T + 32 weeks |

(T = project start date: 2026-08-23)

---

## Checklist before freezing

- [ ] Retrospective pilot success criteria met
- [ ] Power analysis complete with estimated N
- [ ] Prompts frozen and version-tagged
- [ ] Leakage controls validated in pilot
- [ ] ForecastBench team registration complete
- [ ] Cluster compute budget confirmed
- [ ] All co-authors reviewed this document
