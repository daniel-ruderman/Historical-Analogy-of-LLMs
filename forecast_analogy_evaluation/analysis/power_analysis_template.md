# Power Analysis Template

**Version:** 1.0  
**Last updated:** 2026-08-23

Fill this in after the retrospective pilot completes.

---

## Inputs (from pilot)

| Parameter | Symbol | Pilot estimate | Source |
|-----------|--------|---------------|--------|
| Paired Brier difference (analogy − plain) | δ_obs | | `results/aggregate_summary.json` |
| Paired difference SD | σ_Δ | | `results/paired_effects.csv` |
| Coverage (both arms) | c | | run manifest |
| Questions per ForecastBench round | n_round | ~400 binary | ForecastBench docs |

---

## Sample size formula

For a one-sided test at significance α and power 1 − β:

```
N = (z_α + z_β)² × σ²_Δ / δ²
```

Defaults:

- α = 0.05 → z_α = 1.645
- Power = 0.80 → z_β = 0.842
- Minimum detectable effect δ = 0.01 Brier points

---

## Calculation

```
σ_Δ   = [FILL FROM PILOT]
δ     = 0.01
N     = (1.645 + 0.842)² × σ_Δ² / 0.01²
      = [COMPUTE]

Rounds needed = ceil(N / n_round)
              = [COMPUTE]
```

---

## Decision

| Scenario | Rounds | Total questions | Notes |
|----------|--------|----------------|-------|
| Preregistered default | 6 | ~2400 | Fixed stopping rule |
| Power-based estimate | | | |

**Decision:** [Keep 6 rounds / Adjust to N rounds]

**Signed off:** [name, date]

---

## Sensitivity

Repeat calculation for δ = 0.005 (small effect) and δ = 0.02 (large effect).

| MDE (δ) | N needed | Rounds |
|---------|----------|--------|
| 0.005 | | |
| 0.010 | | |
| 0.020 | | |

---

## Notes

- Cluster dependence reduces effective N; the formula above is conservative if σ_Δ
  is computed from paired differences (which already account for question-level pairing).
- If pilot σ_Δ is very large (> 0.15), consider reducing MDE or increasing rounds.
- Do not adjust the preregistered round count after seeing prospective results.
