# FutureEval Replication Plan

**Version:** 1.0  
**Last updated:** 2026-08-23  
**Config:** [`../configs/futureeval_replication.yaml`](../configs/futureeval_replication.yaml)

---

## Purpose

External-validity replication of the ForecastBench findings on
[Metaculus FutureEval](https://www.metaculus.com/futureeval/).

FutureEval complements ForecastBench by:

- Testing on **live, contamination-resistant** questions
- Including **numeric, multiple-choice, and discrete** question formats
- Providing human baselines (Pro Forecasters, community)
- Running seasonal bot tournaments with prize pools

This replication is **secondary** to the ForecastBench confirmatory study. It should
begin only after the ForecastBench protocol is frozen and ideally after initial results
are available.

---

## Why not FutureEval as primary?

| Factor | ForecastBench | FutureEval |
|--------|--------------|------------|
| Static downloadable dataset | Yes | No (live API) |
| Three parallel submission arms | Yes (3 per round) | Requires separate bot identities |
| Binary Brier scoring | Standard | Peer-relative log score |
| Question formats | Binary (+ dataset horizons) | Binary, numeric, MC, discrete |
| Submission window | 24 hours per round | 1.5 hours per question batch |
| Research data access | Open datasets repo | Requires data request |

---

## Recommended tournament

**Seasonal Bot Tournament** (~300–500 questions, 4-month seasons, 3×/year)

- Not MiniBench (auto-generated, noisier, ~60 questions)
- Not Market Pulse (continuous updating, harder to match arms)

Seasons start January, May, September.
See [participate](https://www.metaculus.com/futureeval/participate/) and
[resources](https://www.metaculus.com/notebooks/38928/bot-tournament-resources-page/).

---

## Bot architecture

Use the same three-arm design, implemented as **separate Metaculus bot accounts**:

| Bot | Arm | Notes |
|-----|-----|-------|
| `hal-forecast-plain` | plain | Question only |
| `hal-forecast-delib` | matched_deliberation | Non-analogy deliberation |
| `hal-forecast-analogy` | historical_analogy | Full agentic pipeline |

### Prerequisites (action items)

- [ ] Create three Metaculus bot accounts
- [ ] Request permission for parallel bot identities from Metaculus organizers
- [ ] Request expanded research data access via
      [FutureEval resources](https://www.metaculus.com/notebooks/38928/futureeval-resources-page/)
- [ ] Obtain API tokens for each bot
- [ ] Fork/adapt [metac-bot-template](https://github.com/Metaculus/metac-bot-template)

---

## Scoring differences

FutureEval uses **peer-relative spot scores** (log score minus peer average), not raw Brier.

For replication consistency with ForecastBench:

1. **Primary replication metric:** raw per-question log loss (not peer-relative score).
2. **Descriptive metric:** FutureEval peer score (for comparison with leaderboard).
3. For numeric/MC questions: use Metaculus log-probability / log-density scoring.

Analyze paired differences the same way as ForecastBench (cluster-bootstrap by tournament season).

---

## Question format handling

| Format | Forecast output | Scoring |
|--------|----------------|---------|
| Binary | `p_yes` ∈ [0, 1] | Brier or log loss |
| Multiple choice | `probabilities[]` summing to 1 | Log probability of realized outcome |
| Numeric | `cdf` or `{mean, stdev}` | Log density with platform smoothing |
| Discrete | Same as numeric | Log density |

The initial ForecastBench study uses binary only. FutureEval replication extends to all formats
using the same three-arm logic but format-appropriate forecast schemas.

---

## Leakage considerations

FutureEval-specific protections:

- Human Pro and community forecasts are hidden during bot evaluation
- Submit forecasts within the 1.5-hour window; no post-resolution updates
- Do not copy MetaculusBot (`metac-*`) prompts — use our frozen prompts
- Record exact submission timestamp for each forecast

---

## Timeline

| Step | When |
|------|------|
| Request bot permissions and data access | After ForecastBench preregistration frozen |
| Implement bot adapter | 2–4 weeks |
| Enter next seasonal tournament | Next available season start |
| Score after season resolves | ~4 months after season start |
| Compare with ForecastBench results | After both complete |

---

## Success criteria for replication

Replication "succeeds" if:

- The sign of `Δ_practical` matches ForecastBench (analogy better vs plain)
- The 95% CI overlaps with the ForecastBench CI (consistent effect size)
- Coverage ≥ 90% on all three bots

Replication "fails" if:

- Sign reverses with non-overlapping CIs
- Null result in both benchmarks → report honestly as evidence against the hypothesis

---

## References

- [FutureEval methodology](https://www.metaculus.com/futureeval/methodology/)
- [Bot tournament resources](https://www.metaculus.com/notebooks/38928/bot-tournament-resources-page/)
- [Metaculus bot template](https://github.com/Metaculus/metac-bot-template)
- [forecasting-tools library](https://github.com/Metaculus/forecasting-tools)
