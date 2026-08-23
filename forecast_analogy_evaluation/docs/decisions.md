# Design Decisions Log

Chronological record of evaluation design decisions. Each entry includes rationale so
collaborators and future readers understand *why*, not just *what*.

---

## 2026-08-23 — Move into main repository

**Decision:** Move `forecast_analogy_evaluation/` under `Historical-Analogy-of-LLMs/`.

**Rationale:** Same project, same git repo, easier sharing with collaborators. The
evaluation consumes the parent `agentic_pipeline/` directly via `repo_path: ".."`.

**Protocol impact:** Cluster sbatch paths updated; old cluster path symlinked during
transition so the running pilot job (57471686) could continue.

---

## 2026-08-23 — Project bootstrap (superseded location)

**Decision:** Create `forecast_analogy_evaluation/` as a subdirectory of the main repo.

**Rationale:** Collaborators working on forecasting evaluation use the same checkout.
Originally created as a sibling directory; moved into the repo the same day.

---

## 2026-08-23 — Primary vs secondary hypothesis ordering

**Decision:** Primary = complete analogy pipeline vs plain LLM (practical).
Secondary = analogy content vs matched deliberation (mechanistic).

**Rationale:** First establish whether the system is useful in practice. Only then ask
whether analogies specifically explain the gain. This ordering reflects the project's
practical motivation without sacrificing causal identification (secondary comparison).

---

## 2026-08-23 — Three-arm design with matched deliberation

**Decision:** Include a `matched_deliberation` arm with same model, tool budget, calls,
and output-token cap as the analogy arm.

**Rationale:** Without this control, a win for the analogy pipeline could be attributed
to "thinking longer" or using tools rather than to historical analogies specifically.

**Alternatives considered:**
- Two-arm only (analogy vs plain) — rejected; cannot separate content from compute.
- Four-arm with separate tool-free analogy — deferred; too expensive for initial study.

---

## 2026-08-23 — ForecastBench as primary benchmark

**Decision:** Use ForecastBench for confirmatory evaluation; FutureEval for replication.

**Rationale:**
- ForecastBench provides downloadable question/resolution sets, open evaluation code,
  and allows three submissions per round (matches our three arms).
- FutureEval is live/contamination-resistant but lacks a static downloadable benchmark
  and has tighter bot-submission constraints.

**References:** [ForecastBench](https://www.forecastbench.org/), [FutureEval](https://www.metaculus.com/futureeval/)

---

## 2026-08-23 — Staged evaluation (retrospective then prospective)

**Decision:** Run a retrospective post-cutoff replay for immediate feedback, then a
prospective live study for the confirmatory claim.

**Rationale:** User needs actionable evaluation now but is willing to wait for resolution
on live questions. Retrospective pilot estimates variance and debugs prompts; prospective
study provides contamination-resistant evidence.

---

## 2026-08-23 — Binary questions only for primary analysis

**Decision:** Primary confirmatory analysis uses binary Brier loss only.

**Rationale:** ForecastBench's main scoring is binary Brier; keeps analysis simple and
aligned with the benchmark. Numeric/multiple-choice reserved for FutureEval replication.

---

## 2026-08-23 — qwen3:8b as initial model

**Decision:** Use `qwen3:8b` (local via Ollama) for all roles in the initial evaluation.

**Rationale:** Already validated in the parent repo; runs on cluster GPU (RTX 2080 Ti, ~6.3 GB).
Keeps experimental conditions clean (one model for generation, deliberation, and forecasting).

---

## 2026-08-23 — Separate generation and forecasting prompts

**Decision:** Analogy/deliberation generation and final forecasting are separate LLM calls
with separate prompts.

**Rationale:** Enables auditing of generated packets, reuse across conditions, and clean
attribution of what the forecaster actually saw.

---

## 2026-08-23 — Living documentation as part of the experiment

**Decision:** Maintain protocol, assumptions, leakage, decisions, and experiment logs as
continuously updated documents, not end-of-project writeups.

**Rationale:** User must submit a final report; traceability from day one prevents
post-hoc rationalization and supports collaborator onboarding.

---

## Template for future entries

```
## YYYY-MM-DD — [Short title]

**Decision:** ...

**Rationale:** ...

**Alternatives considered:** ...

**Protocol impact:** [none | amendment required | new arm added]
```
