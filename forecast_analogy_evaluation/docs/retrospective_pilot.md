# Retrospective Pilot Design

**Version:** 1.0  
**Last updated:** 2026-08-23  
**Config:** [`../configs/retrospective_pilot.yaml`](../configs/retrospective_pilot.yaml)

---

## Purpose

Provide **immediate, actionable evaluation** while the prospective study waits for
question resolution. The retrospective pilot is **not** the confirmatory result.

### What the pilot is good for

- Prompt and pipeline debugging
- Detecting large effects (> 0.02 Brier points)
- Estimating paired-loss variance for power analysis
- Validating the three-arm workflow end-to-end
- Testing leakage controls before going live

### What the pilot is NOT good for

- Claiming analogies improve forecasting (contamination risk)
- Publishing as the primary result
- Cherry-picking favorable question subsets

---

## Data source

**ForecastBench archived datasets**

- Questions: [forecastbench-datasets](https://github.com/forecastingresearch/forecastbench-datasets)
- Use resolved question + resolution pairs from rounds **after the model's knowledge cutoff**
- Record exact dataset commit/hash in `data/manifests/dataset_registry.jsonl`

### Question selection procedure

1. Download the latest question set and matching resolution set.
2. Filter to **binary** questions with known resolution.
3. Filter to `question_publish_date > model_knowledge_cutoff`.
4. Verify `t_forecast < resolution_date`.
5. Include **all** remaining questions — no cherry-picking.
6. Record every exclusion in `inclusion_exclusion_ledger.csv` with reason.

---

## Temporal reconstruction

For each question at its original `t_forecast`:

1. Use the **frozen question text and background** from the benchmark snapshot.
2. Do **not** perform live web search against current pages.
3. For analogy generation, restrict Wikipedia search to historical events that
   concluded before `t_forecast` (see [`data_leakage.md`](data_leakage.md)).
4. Cache Wikipedia pages retrieved during the pilot in `data/cache/` with timestamps.

---

## Arms

### Main arms (same as confirmatory)

1. `plain`
2. `matched_deliberation`
3. `historical_analogy`

### Diagnostic arms (pilot only)

4. **`shuffled_analogy`** — Analogy packet from a different question in the same
   ForecastBench category (market vs dataset). Same token budget. Tests whether
   *any* structured historical content helps or only *relevant* content.

5. **`analogy_name_only`** — Only the winning analogy event name (one line), no
   similarities/differences/counterexamples. Tests whether structured comparison matters.

Diagnostic arms are analyzed **exploratory only**.

---

## Analysis priority in the pilot

Report results in this order:

1. **Primary contrast:** `historical_analogy` vs `plain` (practical effect)
2. **Secondary contrast:** `historical_analogy` vs `matched_deliberation` (mechanistic)
3. Diagnostic contrasts (shuffled, name-only)
4. Paired-loss variance estimate → feed into power analysis

---

## Pilot success criteria

The pilot is "successful" (ready for prospective preregistration) when:

- [ ] All three main arms produce valid forecasts on ≥ 90% of eligible questions
- [ ] Leakage checks pass (zero L2/L3 incidents, or all incidents documented)
- [ ] Analogy packets are saved and auditable for ≥ 90% of analogy-arm questions
- [ ] Paired-loss variance is estimated with CI width < 0.005
- [ ] No unresolved protocol ambiguities remain

The pilot does **not** need to show a positive effect to proceed to the prospective stage.

---

## Sample size

Start with **one complete ForecastBench round** (~500 questions, binary subset likely 400+).
If compute is limited, use a **stratified random sample** of 100 questions (50 market + 50
dataset) — record the sampling seed and document as a pilot limitation.

---

## Compute estimate

Per question (approximate, based on agentic smoke test):

| Step | LLM calls | Time (qwen3:8b, GPU) |
|------|-----------|---------------------|
| Analogy generation | ~140 | ~30–60 min |
| Deliberation generation | ~10–20 | ~5 min |
| Forecasting (×3 arms) | 3 | ~1 min |
| **Total per question** | ~160 | ~35–65 min |

For 100 questions: ~60–110 GPU-hours. Run on cluster via Slurm (see cluster-usage skill).

Batching strategy: generate all analogy packets first (resumable), then run all forecasts.
