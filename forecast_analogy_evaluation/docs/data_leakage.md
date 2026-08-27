# Data Leakage Policy

**Version:** 1.0  
**Last updated:** 2026-08-23

Data leakage is the primary threat to validity in this evaluation. This document defines
how we prevent, detect, and report leakage at every stage.

---

## 1. Leakage types

| Type | Description | Severity |
|------|-------------|----------|
| **L1 — Outcome leakage** | Model or search corpus contains the resolved answer | Critical |
| **L2 — Forecast leakage** | Analogy search retrieves crowd forecasts, market prices, or discussion of the target question | Critical |
| **L3 — Temporal leakage** | Sources postdating `t_forecast` enter the analysis | High |
| **L4 — Training contamination** | Question/outcome appeared in model pretraining data | Medium (unavoidable; mitigated by post-cutoff selection) |
| **L5 — Pipeline leakage** | Later condition sees earlier condition's output within the same question | High |
| **L6 — Analyst leakage** | Exclusion or protocol changes made after seeing condition outcomes | High |

---

## 2. Temporal cutoff policy

Every question has a **forecast timestamp** `t_forecast` (when the prediction is made).

### Rule T1 — No future information

No source with publication or last-modified date **after** `t_forecast` may enter any arm.

### Rule T2 — Historical corpus only for analogy search

Analogy-generation searches may retrieve:

- Wikipedia articles about **historical events that concluded before** `t_forecast`
- Reference works, encyclopedias, academic histories with known publication dates ≤ `t_forecast`

Analogy-generation searches may **NOT** retrieve:

- The ForecastBench / Metaculus / Polymarket / Kalshi question page itself
- Pages discussing the specific forecast question or its likely outcome
- Crowd forecasts, prediction-market prices, or aggregator sites for the target question
- News articles published after `t_forecast` about events relevant to the question

### Rule T3 — Post-cutoff question selection (retrospective)

Include only questions where:

```
question_publish_date > model_knowledge_cutoff
AND
t_forecast < resolution_date
```

Record `model_knowledge_cutoff` per model in the run manifest. For `qwen3:8b`, use the
documented Qwen3 training cutoff (verify from model card at run time).

### Rule T4 — Frozen research packet (if contemporary retrieval is used)

If any arm uses web search on contemporary sources:

1. Generate **one** research packet at `t_forecast` with temporal filters.
2. Share that **identical packet** across all three main arms.
3. Only the reasoning treatment (plain vs deliberation vs analogy) may differ.

---

## 3. Condition isolation

### Rule C1 — Fresh context per arm

Each arm for the same question runs in a **separate conversation context**. No arm may
see another arm's forecast or analysis packet.

### Rule C2 — Randomized order

Process arms in random order within each question to avoid order effects.

### Rule C3 — Separate generation and forecasting

Analogy/deliberation packets are generated and saved **before** the forecasting call.
The forecaster prompt contains only the packet — not the generation conversation history.

---

## 4. Source restrictions for Wikipedia search

The agentic pipeline uses Wikipedia search/lookup. Apply these filters:

| Allowed | Blocked |
|---------|---------|
| Historical event articles (e.g., "Velvet Revolution") | ForecastBench question titles |
| Biographical/historical context predating `t_forecast` | "2026 election forecast" style pages |
| Verified past event outcomes | Pages whose primary topic is the target question |

### Implementation checklist

- [ ] Search queries must not contain the exact ForecastBench question text
- [ ] Lookup titles must not match the forecast question title
- [ ] Log every search query and returned title for post-hoc audit
- [ ] Flag any result whose title overlaps > 80% with the question title (Jaccard on tokens)

---

## 5. Contamination checks

Run these checks **before** scoring:

### Check 1 — Timestamp audit

Verify `forecast_timestamp < resolution_date` for every row in `forecasts.jsonl`.

### Check 2 — Source date audit (retrospective)

For every retrieved URL in `tool_traces.jsonl`, verify cached page date ≤ `t_forecast`.
Flag violations as `contamination_detected` in the inclusion ledger.

### Check 3 — Probability sanity

Flag forecasts where `p_yes` is exactly 0.0 or 1.0 (overconfidence) or exactly 0.5
 across > 80% of questions in a run (possible default/refusal pattern).

### Check 4 — Cross-arm contamination

Verify no `analysis_packet` from question A appears in question B's forecaster prompt
(unless deliberately shuffled in the diagnostic arm).

### Check 5 — Post-hoc exclusion audit

Verify all exclusions in the inclusion ledger were recorded before any condition's
Brier loss was computed for that question.

---

## 6. Handling leakage incidents

When leakage is detected:

1. **Do not silently drop** the question — record it in the inclusion ledger with reason
   `contamination_detected` and a description of the incident.
2. Log the incident in [`experiment_log.md`](experiment_log.md) with date, question ID,
   leakage type (L1–L6), and remediation.
3. Re-run affected questions only if the leakage source is fixed and documented in
   [`decisions.md`](decisions.md).
4. Report the count of leakage incidents in the final report (transparently).

---

## 7. Prospective-specific protections

For live ForecastBench rounds:

- Generate all forecasts **before** any question in the round resolves.
- Timestamp and hash (`sha256`) every artifact at generation time.
- Store artifacts in immutable run directories — never overwrite.
- Do not re-query search engines after resolution for "better" analogies.

---

## 8. What we cannot fully prevent

| Risk | Mitigation | Accepted? |
|------|------------|-----------|
| Model pretraining on question text | Post-cutoff selection; prospective stage | Partially |
| Wikipedia content updated after `t_forecast` | Use cached snapshots where available | Partially |
| Non-deterministic model behavior | Pin model version; record seeds | Yes |
| Subtle prompt leakage via question background | Background is identical across arms by design | Yes |

---

## 9. Incident log

| Date | Question ID | Leakage type | Description | Action |
|------|------------|--------------|-------------|--------|
| | | | | |

Update this table whenever an incident occurs.
