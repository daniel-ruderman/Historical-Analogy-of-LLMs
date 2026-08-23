# Final Report Outline

**Version:** 1.0  
**Last updated:** 2026-08-23

This outline maps every report section to its source documents and artifacts.
Update the "Status" column as sections are completed.

---

## 1. Abstract

**Status:** Not started

One paragraph: primary finding (analogy pipeline vs plain), secondary finding (mechanistic),
benchmark, effect size, and main limitation.

**Sources:** `results/aggregate_summary.json`

---

## 2. Introduction

### 2.1 Motivation

- Historical analogies are widely used in policy and intelligence analysis
- LLMs can generate analogies (parent repo) but does this improve forecasting?
- Gap: no preregistered evaluation of analogy-augmented forecasting

**Sources:** parent repo `PROJECT.md` §12, this protocol

### 2.2 Research questions

**Sources:** [`evaluation_protocol.md`](evaluation_protocol.md) §1

### 2.3 Contributions

- Three-arm paired design separating practical and mechanistic effects
- Retrospective pilot + prospective ForecastBench + FutureEval replication
- Open evaluation framework in [`forecast_analogy_evaluation/`](../)

---

## 3. Background

### 3.1 Historical analogy acquisition (parent repo)

Brief summary of the agentic pipeline (Generate/Search, Critic, Anti-Analogy, Judge, Summarizer).

**Sources:** [`PROJECT.md`](../../PROJECT.md), smoke test results

### 3.2 Forecasting benchmarks

- ForecastBench: design, scoring, submission
- FutureEval: live evaluation, bot tournaments
- Why ForecastBench is primary

**Sources:** [`decisions.md`](decisions.md), [`futureeval_replication.md`](futureeval_replication.md)

---

## 4. Methods

### 4.1 Treatment arms

**Sources:** [`evaluation_protocol.md`](evaluation_protocol.md) §2, `prompts/`

### 4.2 Two-stage architecture

Diagram: question → {plain, deliberation, analogy} → forecaster → p(YES)

**Sources:** [`evaluation_protocol.md`](evaluation_protocol.md) §3

### 4.3 Question eligibility and sample

**Sources:** [`evaluation_protocol.md`](evaluation_protocol.md) §4,
`data/manifests/inclusion_exclusion_ledger.csv`

### 4.4 Model and compute

**Sources:** run manifests, [`decisions.md`](decisions.md)

### 4.5 Data leakage controls

**Sources:** [`data_leakage.md`](data_leakage.md)

### 4.6 Endpoints and analysis

**Sources:** [`evaluation_protocol.md`](evaluation_protocol.md) §6–7,
[`../analysis/analysis_spec.md`](../analysis/analysis_spec.md)

### 4.7 Analogy quality audit

**Sources:** [`evaluation_protocol.md`](evaluation_protocol.md) §9

---

## 5. Results

### 5.1 Retrospective pilot

- Sample description (N, domains, horizons)
- Primary Δ (analogy vs plain) with CI
- Secondary Δ (analogy vs matched deliberation) with CI
- Diagnostic arms (exploratory)
- Coverage and failure rates

**Sources:** `results/paired_effects.csv`, [`experiment_log.md`](experiment_log.md)

**Limitation callout:** retrospective ≠ confirmatory

### 5.2 Prospective ForecastBench (confirmatory)

- Rounds completed, resolution status
- Primary endpoint with CI and p-value
- Secondary endpoint (Holm-adjusted)
- Coverage, cost

**Sources:** `results/aggregate_summary.json`, [`prospective_preregistration.md`](prospective_preregistration.md)

### 5.3 Subgroup analyses (exploratory)

- Market vs dataset
- Horizon short vs long
- Analogy relevance and quality moderation

**Sources:** `results/subgroup_effects.csv`

### 5.4 FutureEval replication

- Consistency with ForecastBench
- Format extensions (numeric, MC)

**Sources:** FutureEval run manifests, [`futureeval_replication.md`](futureeval_replication.md)

### 5.5 Analogy quality audit results

- Factuality, relevance, diversity distributions
- Correlation between quality and Δ

**Sources:** quality audit ratings (to be collected)

### 5.6 Case studies

- 2–3 examples where analogies helped
- 2–3 examples where analogies hurt (hallucinations, surface similarity)

**Sources:** `runs/*/analysis_packets.jsonl`, smoke test "Roman Revolution of 1968"

---

## 6. Discussion

### 6.1 Summary of findings

**Sources:** §5 aggregate

### 6.2 Practical vs mechanistic interpretation

- If primary+secondary both positive → analogies are useful and causally implicated
- If primary positive, secondary null → deliberation/tools drive the gain
- If both null → analogies don't help on average (subgroups may differ)

**Sources:** [`assumptions_and_limitations.md`](assumptions_and_limitations.md) §4

### 6.3 Comparison with prior work

- ForecastBench leaderboard baselines
- FutureEval bot tournament results
- Historical analogy literature

### 6.4 Limitations

**Sources:** [`assumptions_and_limitations.md`](assumptions_and_limitations.md) §3

### 6.5 Future work

- Multi-analogy ensembles
- Stronger verification
- Frontier model evaluation
- Domain-specific corpora beyond Wikipedia

---

## 7. Conclusion

**Status:** Not started

---

## 8. References

- ForecastBench paper and docs
- FutureEval / Metaculus
- Parent repo paper
- Relevant forecasting literature

---

## 9. Appendices

### A. Frozen prompts (v1)

**Sources:** `prompts/`

### B. Inclusion/exclusion ledger

**Sources:** `data/manifests/inclusion_exclusion_ledger.csv`

### C. Leakage incident log

**Sources:** [`data_leakage.md`](data_leakage.md) §9

### D. Design decisions log

**Sources:** [`decisions.md`](decisions.md)

### E. Experiment log

**Sources:** [`experiment_log.md`](experiment_log.md)

### F. Preregistration (frozen version)

**Sources:** [`prospective_preregistration.md`](prospective_preregistration.md)

### G. Analysis code

**Sources:** `analysis/`

---

## Traceability matrix

| Claim | Analysis | Run | Manifest | Limitation |
|-------|----------|-----|----------|------------|
| Primary Δ | `analysis/scoring.py` | `runs/<id>/` | `manifest.json` | retrospective contamination |
| Secondary Δ | same | same | same | matched-deliberation token match |
| Subgroup | `analysis/subgroups.py` | same | same | exploratory, low power |
| Quality moderation | manual audit | same | `analysis_packets.jsonl` | judge subjectivity |

Update this matrix as results become available.
