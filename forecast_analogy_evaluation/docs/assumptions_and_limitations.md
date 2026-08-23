# Assumptions and Limitations

**Version:** 1.0  
**Last updated:** 2026-08-23

This document records scientific assumptions **before** analysis and known limitations
**as they are discovered**. Update it continuously — the final report draws from here.

---

## 1. Assumptions

### A1 — Forecasting is a valid test of analogy usefulness

We assume that improved probabilistic accuracy on real forecasting questions is a
meaningful operationalization of “better inference.” This does not capture all forms of
historical reasoning (e.g., qualitative policy advice, counterfactual explanation).

### A2 — The plain arm is a fair baseline

The plain arm receives the same question text, background, resolution criteria, and
forecast timestamp as the treatment arms. We assume this represents what a competent
user would provide to an LLM without analogies.

### A3 — Matched deliberation controls for compute

The matched-deliberation arm uses the same model, tool budget, number of tool calls,
and output-token cap as the analogy arm. We assume this sufficiently controls for
“thinking longer” without guaranteeing perfect token-level matching.

### A4 — Temporal reconstruction is feasible (retrospective)

For the interim pilot, we assume we can reconstruct the information landscape at each
original forecast timestamp using frozen benchmark snapshots and a historical corpus.
Perfect reconstruction is not achievable; see limitations below.

### A5 — Binary Brier is the right primary metric

We assume binary Brier loss on clipped probabilities is the standard, interpretable
metric for ForecastBench-style questions. It penalizes both miscalibration and
overconfidence symmetrically.

### A6 — Question independence within clusters

For uncertainty estimation we treat questions as independent **conditional on cluster**
(ForecastBench round, underlying event/series). Related horizons from the same series
are explicitly clustered.

### A7 — Analogy pipeline output is stable for a frozen config

We assume that re-running the agentic pipeline with the same model, prompts, and seeds
(on a given question) produces sufficiently similar packets for reproducibility.
Non-determinism from sampling is recorded but not eliminated.

### A8 — Wikipedia is an adequate historical corpus

The agentic pipeline verifies analogies via Wikipedia search/lookup. We assume this
is sufficient for fact-checking historical events but not for all domain-specific
forecast questions (e.g., financial series).

---

## 2. Threats to validity

| Threat | Mitigation | Residual risk |
|--------|------------|---------------|
| **Outcome leakage** (model saw answer in training) | Post-cutoff questions only; prospective stage | Medium in retrospective |
| **Information leakage** (search finds current forecasts) | Source restrictions in [`data_leakage.md`](data_leakage.md) | Low if enforced |
| **Construct validity** (Brier ≠ “better reasoning”) | Secondary metrics + quality audit | Medium |
| **Confound: extra tokens** | Matched deliberation arm | Low–medium |
| **Confound: tool use** | Same tool budget across matched arms | Low |
| **Selection bias** | Inclusion ledger before outcome inspection | Low if enforced |
| **Cluster dependence** | Cluster-robust bootstrap / mixed model | Low if modeled |
| **Model drift** | Pin model version; log every run | Medium over long studies |
| **Benchmark non-representativeness** | FutureEval replication | Medium |

---

## 3. Known limitations

### 3.1 Retrospective pilot

- **Not contamination-proof.** Even post-cutoff questions may appear in model training
  data, and reconstructed corpora may include information that was not actually available
  at `t_forecast`.
- **Cannot support the primary scientific claim.** Useful for engineering, power analysis,
  and detecting large effects only.
- **Resolution timing.** Some questions resolve long after the forecast window; the pilot
  uses already-resolved subsets which introduces additional hindsight bias in corpus curation.

### 3.2 Prospective study

- **Long wait for resolution.** Market questions may take months; dataset questions resolve
  faster but are more base-rate-friendly.
- **Three-arm limit.** ForecastBench allows three submissions per team per round — exactly
  matches our three main arms but leaves no room for diagnostic arms without a protocol amendment.
- **50-day leaderboard delay.** Official ForecastBench scoring stabilizes ~50 days after
  forecast; we score locally as soon as resolutions arrive.
- **GPU/compute constraints.** The agentic pipeline is expensive (~140 LLM calls per
  question for analogy generation alone); full 500-question rounds require cluster scheduling.

### 3.3 Analogy pipeline

- **Hallucinated analogies.** Smoke test showed a fabricated “Roman Revolution of 1968”
  candidate that passed initial verification. Quality audit is essential.
- **Surface similarity trap.** Models may pick famous events with shared vocabulary but
  different causal structure (exactly what MDS penalizes — but MDS is not the forecast metric).
- **Single analogy winner.** The pipeline selects one winning analogy; multi-analogy
  ensembles are not tested.

### 3.4 Generalization

- **English only.** Wikipedia search is English-language.
- **Binary focus.** Primary analysis excludes numeric and multiple-choice questions
  (reserved for FutureEval replication).
- **One model initially.** Results may not transfer to frontier API models.

---

## 4. Interpretation boundaries

1. A **retrospective win** is promising engineering evidence, not a contamination-free result.
2. A **primary win** (analogy vs plain) means the system is practically useful; it does
   **not** prove analogies caused the improvement.
3. A **secondary win** (analogy vs matched deliberation) supports the mechanistic claim.
4. A **null primary result** does not prove analogies are useless — they may help in
   subgroups (exploratory) or with better pipeline quality.
5. A **null secondary result** with a positive primary result suggests the gain comes from
   deliberation/tool use rather than analogy content specifically.
6. **Subgroup findings** require independent replication; do not elevate to headline claims.

---

## 5. Limitation log

Record new limitations here as they are discovered during experiments.

| Date | Limitation | Impact | Action |
|------|-----------|--------|--------|
| 2026-08-23 | Analogy pipeline can hallucinate event names | May produce harmful forecasts | Quality audit + verification hardening |
| 2026-08-23 | Retrospective corpus reconstruction is imperfect | Inflates or deflates pilot effects | Use pilot for variance only; confirm prospectively |
| | | | |
