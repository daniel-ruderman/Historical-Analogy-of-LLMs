# Experiment Log

Chronological record of every pilot and live run — **including failures**.

| Date | Run ID | Stage | Config | Questions | Status | Notes |
|------|--------|-------|--------|-----------|--------|-------|
| 2026-08-23 | (infra) | — | — | — | complete | Moved to Historical-Analogy-of-LLMs/forecast_analogy_evaluation/ |
| 2026-08-23 | dryrun_smoke_003 | setup | retrospective_pilot | 2 | complete | Dry-run pipeline validation (mock LLM) |
| 2026-08-23 | pilot_pilot_57471686 | retrospective_pilot | retrospective_pilot | 20 | running | Full pilot; Slurm job 57471686, 24h wall time |

---

## Run: pilot_smoke_57457642

- **Date:** 2026-08-23
- **Stage:** retrospective_pilot (smoke)
- **Config:** configs/retrospective_pilot.yaml
- **Model:** qwen3:8b on RTX 2080 Ti (cluster job 57463523)
- **Questions:** 2 / 330 eligible (ForecastBench 2024-07-21 round)
- **Arms:** plain, matched_deliberation, historical_analogy
- **Status:** success (100% coverage all arms)
- **Artifacts:** runs/pilot_smoke_57457642/, results/pilot_smoke_57457642/
- **Primary Δ (analogy − plain Brier):** −0.07 (analogy slightly better; n=2, not significant)
- **Secondary Δ (analogy − delib Brier):** +0.08 (analogy slightly worse; n=2)
- **Failures:** first attempt (job 57457642) failed with `set_settings()` TypeError; fixed and resumed
- **Deviations from protocol:** smoke mode (1 refinement round, 2 ReAct steps)
- **Next steps:** scale to 20–100 questions for variance estimate; then preregister live rounds


Copy this block for each run:

```
## Run: <run_id>

- **Date:** YYYY-MM-DD
- **Stage:** retrospective_pilot | prospective_live | futureeval_replication | dry_run
- **Config:** configs/<name>.yaml
- **Model:** qwen3:8b (or other)
- **Questions:** N eligible / M total
- **Arms:** plain, matched_deliberation, historical_analogy [, diagnostic arms]
- **Status:** success | partial | failed
- **Artifacts:** runs/<run_id>/
- **Primary Δ (analogy − plain):** [fill after scoring]
- **Secondary Δ (analogy − delib):** [fill after scoring]
- **Failures:** [parse errors, refusals, leakage incidents]
- **Deviations from protocol:** [none | describe]
- **Next steps:** [what to fix or run next]
```

---

## Failures and incidents

Record every failure here, even if the run was later re-attempted.

| Date | Run ID | Type | Description | Resolution |
|------|--------|------|-------------|------------|
| 2026-08-23 | pilot_smoke_57457642 | other | First job failed: set_settings() missing argument | Fixed engine.py; resumed via job 57463523 |

Types: `parse_failure`, `leakage`, `oom`, `timeout`, `refusal`, `verification_fail`, `other`
