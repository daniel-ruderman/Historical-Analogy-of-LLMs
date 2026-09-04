# CLAUDE.md — orientation for coding agents

Read this first. It is the map of the repository, the rules that must not be broken,
and the current state of the research.

- **PROJECT.md** — the research/technical deep dive (metric formulas, prompt provenance,
  deviations from the paper).
- **PROJECT_GUIDE.txt** — the practical human guide (setup, commands, troubleshooting).
- **This file** — what an agent needs to work here safely.

---

## 1. What this project is

A research fork of the code for **"Past Meets Present: Creating Historical Analogy with
Large Language Models"** (Li et al.). Task: given a contemporary event, find a past event
that is genuinely analogous — matching deep structure (**Topic / Background / Process /
Result**), not surface wording.

Authors of this extension: Daniel Ruderman, Yuval Rom.

**Research question:** does an agentic pipeline (search + criticism + counterexamples)
produce better historical analogies than plain LLM prompting?

Three strictly separate bodies of code:

| | Location | Rule |
|---|---|---|
| 1. Original paper code | `framework/`, `evaluation.py` | **Never modify.** Reference only. |
| 2. Our provider-neutral baselines | `gemini_baselines/` | Must stay *faithful* to the paper. |
| 3. Our new agentic method | `agentic_pipeline/` | This is the contribution. |

---

## 2. Hard invariants (violating these breaks the research)

1. **Never edit `framework/**` or `evaluation.py`.** They are the authors' implementation,
   kept for reference. Every change goes in our own modules. `git diff HEAD -- framework
   evaluation.py` must stay empty.
2. **Baselines must not be improved.** `gemini_baselines/` reproduces the paper's prompts
   (copied verbatim into `gemini_baselines/prompts.py`), candidate counts (10 / 10 / 5,
   top-10 retrieval), temperature (0.1), and control flow. Do **not** add search, critics
   or better prompts there — that would make the comparison unfair. Improvements belong in
   `agentic_pipeline/`.
3. **The Final Judge is NOT an agent.** No tools, no ReAct loop, no candidate generation.
   It is a single ranking call after the loop. Never call it "Judge agent" or give it a
   search tool. The three agents are Generate/Search, Critic, Anti-Analogy.
4. **Never hardcode a model name** outside `hal/project_defaults.py`. Algorithms call
   `get_llm(role=...)` and never name a vendor or model.
5. **No API keys in tracked files.** Secrets live only in the git-ignored `.env`.
   `hal/project_defaults.py` raises at import if a secret key is ever added there.
6. **Never fabricate a score.** A failed call is recorded with a status, not a number.
7. **The evaluation metric is the paper's.** Do not invent weights or thresholds.

---

## 3. Stack

- Python 3.13 on the laptop (3.10 on the old desktop; both fine), Windows
  (use `py`, not `python`). No framework — plain classes/functions.
- `google-genai` (the current SDK: `from google import genai`), `requests`, `numpy`,
  `nltk`, `python-dotenv`, `pytest`. See `requirements_project.txt`.
- Deliberately **no** LangChain / LangGraph / CrewAI — the agent loop is ~230 readable
  lines in `agentic_pipeline/react.py` (234) so it can be inspected and modified for research.
- Wikipedia via the MediaWiki API (free) as the agents' search backend.

---

## 4. Architecture

```
hal/                     shared infrastructure — NO research logic lives here
  project_defaults.py    THE tracked shared defaults (model, rounds, …)   <- edit to change
  config.py              env var > project default > code fallback
  providers/
    base.py              LLMProvider / EmbeddingProvider / SearchProvider  (the abstraction)
    gemini.py            Gemini + Gemma implementation (API)
    local.py             LocalLLMProvider — Ollama / LM Studio / llama.cpp  <- CURRENT
    search.py            WikipediaSearchProvider
    mock.py              fakes used by every test
    caching.py           CachedEmbeddingProvider
    factory.py           registry: get_llm(role=…) / get_embedding_provider() / …
  schemas.py             dataclasses passed between components
  cache.py, retry.py, wiki.py, json_utils.py, text_similarity.py, vector.py, io_utils.py

gemini_baselines/        the paper's 6 methods, provider-neutral (faithful)
  prompts.py             the paper's prompts, verbatim
  evaluation_mds.py      our port of evaluation.py — MDS + Pass@1
  direct_retrieval / twostage_retrieval / direct_generation /
  twostage_generation / summary_generation / reflection_generation

agentic_pipeline/        OUR method
  generate_search_agent.py   agent 1 — proposes 5–10 candidates, can search
  critic_agent.py            agent 2 — scores 4 dims 1–4, finds weaknesses
  anti_analogy_agent.py      agent 3 — hunts counterexamples / opposite outcomes
  final_judge.py             ranking component (NOT an agent) + heuristic fallback
  final_summarizer.py        explains the winner (plain LLM call)
  canonical_names.py         resolves the winner to the Wikipedia title it denotes,
                             and skips a candidate that IS the input event
  pipeline.py                the refinement loop + CLI
  react.py                   the think → search → observe loop

automatic_evaluation/    scores all 7 methods; contains NO method logic
  methods.py             maps a method name to the EXISTING implementation
  runner.py              generate / score / resume / aggregate / CSV.
                         Results land in results/automatic_evaluation_<model>/ derived
                         from the models in use; rows are deduped on
                         (dataset, method, index); pooling different models, settings or
                         tokenizers is flagged as "MORE THAN ONE condition"

examples/run_all_methods.py           run methods, inspect answers
examples/run_automatic_evaluation.py  score them with the paper's metric
tests/                                270 tests, all offline (mock providers)
```

**Flow:** input event → Generate/Search agent → candidates → *[Critic + Anti-Analogy →
feedback → Generate/Search revises]* × N rounds → Final Judge ranks → Final Summarizer
explains.

**Metric (from the paper, do not change):**
```
per dimension:  All_d = Abs_d(1–4, LLM judge) × max(α − Lit_d(Jaccard), 0)
MDS = 0.5·T_All + 1·B_All + 2·P_All + 2·R_All        α = 0.35
```
Pass@1 (popular set only): Wikipedia *search* result sets of the produced and reference
names must intersect — not string equality.

---

## 5. Configuration

Resolution order: **environment / `.env`** → **`hal/project_defaults.py`** (tracked, shared
via git) → code fallback in `hal/config.py`.

- Shared research settings → `hal/project_defaults.py` (committed, partner gets them).
- Secrets and personal overrides → `.env` (git-ignored, never shared).
- `hal.config.config_source("LLM_MODEL")` reports where a value came from.

Current tracked defaults: `LLM_PROVIDER=local`, `LLM_MODEL=qwen3:8b`,
`EMBEDDING_MODEL=gemini-embedding-001`, `REFINEMENT_ROUNDS=2`, `MAX_CANDIDATES=8`,
`MAX_OUTPUT_TOKENS=4096`, `WIKI_REQUEST_DELAY=0.2`.

`WIKI_REQUEST_DELAY` paces MediaWiki calls at ~5/s and is deliberately separate from
`REQUEST_DELAY`, which also paces the LLM — a local Ollama needs no pacing. Raise it if you
ever see `! wikipedia lookup failed` during a run — see *"A Wikipedia lookup that FAILS must
never be cached"* in section 8.

Per-role models (`GENERATOR_MODEL`, `CRITIC_MODEL`, `ANTI_ANALOGY_MODEL`, `JUDGE_MODEL`,
`SUMMARIZER_MODEL`, `BASELINE_MODEL`, `EVALUATION_MODEL`) all inherit `LLM_MODEL` unless set.
Per-role **providers** exist too (`GENERATOR_PROVIDER` … `EVALUATION_PROVIDER`), inheriting
`LLM_PROVIDER`. That is how you generate on a local server while judging with an API model —
`LLM_PROVIDER=local LLM_MODEL=qwen3:8b EVALUATION_PROVIDER=gemini EVALUATION_MODEL=…`.
The model that *produces* an analogy and the model that *judges* it are deliberately separate.

---

## 6. Commands

```bash
py -m pytest tests -q                                            # 270 tests, no API quota
py examples/run_all_methods.py --smoke                           # quick sanity check
py examples/run_all_methods.py --dry-run                         # fake providers, no key
py examples/run_automatic_evaluation.py --dataset popular --methods agentic --smoke
py examples/run_automatic_evaluation.py --dataset popular --methods all
py examples/run_automatic_evaluation.py --dataset general --methods all --resume
```

`--resume` skips example/method pairs already scored and reuses saved answers.
`--generate` / `--evaluate` split answer production from scoring.

---

## 7. Current state (verified 2026-08-26)

All 270 tests pass.

### The current condition: `qwen3:8b`, popular (20 examples)

`results/automatic_evaluation_qwen3_8b/`. Generated **and judged** by `qwen3:8b`.

| method | MDS | sd | Pass@1 | scored |
|---|---|---|---|---|
| **agentic** | **3.53** | **0.52** | 0.20 | 20/20 |
| summary_generation | 3.50 | 0.60 | 0.40 | 20/20 |
| direct_retrieval | 3.49 | 0.61 | 0.45 | 20/20 |
| twostage_generation | 3.46 | 0.99 | 0.50 | 20/20 |
| twostage_retrieval | 3.45 | 0.78 | 0.65 | 20/20 |
| direct_generation | 3.42 | 0.59 | 0.35 | 20/20 |
| reflection_generation | 3.42 | 0.77 | 0.20 | 15/20 |

**Read this table with the statistics, not the ordering.** The spread between first and
last is 0.109 while the within-method sd is ~0.7: every paired comparison of agentic
against a baseline spans zero (widest `+0.11 [-0.13, +0.35]`). At n=20 the methods are
**indistinguishable**, and "agentic is first" is not a result. Do not claim otherwise
without the general set.

What *is* real is the variance: agentic was sd 0.997 with a 0.00 worst case before the
2026-08-26 fixes, and is sd 0.524 with no zeroed dimensions after. That makes the general
run more sensitive, which matters more than the 0.12 the mean moved.

qwen3 judges harder than gemma (13/24 agreement on identical pairs, and it never awarded a
4), so these numbers sit systematically below the gemma tables below. That gap is judge
severity, not method quality — never compare across the two.

### The older condition: `gemma-4-31b-it`

`results/automatic_evaluation_gemma_4_31b/`, produced **and judged** by `gemma-4-31b-it`.
Kept as a record; **not comparable** with the table above, and produced before the
Wikipedia cache bug and the answer-parsing bugs were found.

**Popular (20 examples):**

| method | MDS | Pass@1 | scored |
|---|---|---|---|
| direct_retrieval | 3.47 | 0.40 | 20/20 |
| twostage_retrieval | 3.93 | 0.55 | 20/20 |
| direct_generation | 3.95 | 0.60 | 20/20 |
| twostage_generation | 3.10 | 0.10 | 18/20 |
| summary_generation | 3.71 | 0.40 | 20/20 |
| reflection_generation | 4.06 | 0.35 | 20/20 |
| **agentic** | **4.11** | 0.25 | 17/20 |

**General (160 examples):**

| method | MDS | scored |
|---|---|---|
| direct_retrieval | 3.16 | 160/160 |
| twostage_retrieval | 3.65 | 159/160 |
| direct_generation | 3.97 | 150/160 |
| twostage_generation | 3.46 | 116/160 |
| summary_generation | 3.97 | 150/160 |
| reflection_generation | 4.18 | 159/160 |
| **agentic** | 4.03 | **37 scored / 60 attempted** ← incomplete |

---

## 8. Open problems

0. **n=20 cannot answer the research question.** Every method on popular is within 0.11 MDS
   of every other, against a within-method sd of ~0.7. The general set (160) is the run that
   decides anything. Treat any popular-set ordering as noise.
1. **The refinement loop is nearly inert — the biggest open question.** On the 2026-08-23
   run: the final winner was already in the round-1 candidate set in **18/20** examples, the
   candidate set never changed at all in **8/20**, and refinement introduced the winner
   twice (once good, once bad). Only 1 parse error, so it is *choosing* not to change — and
   `GENERATE_REVISE` tells it to ("Do not replace a candidate merely because it was
   criticised"). Strip the loop and the pipeline is essentially `twostage_generation`, which
   is what the near-identical scores suggest. **The decisive experiment is `--rounds 0`**,
   and it should be run now that the Critic finally has a criterion it can act on
   (distinctness, added 2026-08-26) — before that it had no reason to reject anything.
2. **The agentic general run is incomplete** — only 60/160 examples attempted, 37 scored,
   versus ~160 for the baselines. Its MDS is **not yet comparable**. Note that was the
   gemma condition; the general set has not been run at all under `qwen3:8b`.
3. **Agentic scores well on MDS and poorly on Pass@1** (qwen3: 3.53 with Pass@1 0.20,
   against twostage_retrieval's 3.45 / 0.65; the gemma run showed the same shape). Plausibly
   real: MDS rewards structural, non-literal analogies while Pass@1 rewards the canonical
   textbook answer — "Velvet Revolution" for the Arab Spring is arguably better than the
   reference and scores zero. Needs investigation before any claim, and Pass@1 at n=20 has
   error bars of roughly ±0.21.
4. **Unscorable answers skew the averages.** `twostage_generation` lost 44/160 on general;
   `no_analogy` + `unresolved_event` + 21 errors total 187 rows in general. Methods with
   different `n` are not directly comparable.
5. **Pass@1 is loose** — Wikipedia search sets can intersect for merely related events
   (e.g. "Revolutions of 1848" counted as a hit for "Revolutions of 1989"). Inherited from
   the paper's code; kept for faithfulness, but do not over-read it.
6. **Gemma 4 is a poor fit for the agentic pipeline.** On the general set it produced
   *no answer at all* for 34 of 60 attempted examples ("no candidates were produced",
   plus 5 "unparseable model output"), aborting before the refinement loop — so the
   agentic MDS of 4.03 rests on 37 survivors and is likely survivorship-biased. It is
   also slow (~10 min per agentic example) because thinking tokens count against
   `max_output_tokens`. **This is the main driver of the planned move to local models**
   (section 9). The ReAct loop's strict one-JSON-object-per-turn contract is the likely
   friction point; whatever model replaces Gemma should be checked against it first.
7. **Free-tier daily cap is per project per model** (quotaId
   `GenerateRequestsPerDayPerProjectPerModel-FreeTier`). Now largely moot: generation and
   judging are local and unlimited, and only the two retrieval baselines still call the
   Gemini API, for embeddings on a separate quota. Still relevant if anyone switches a role
   back to an API model — and never mix models within one experiment.
8. **Not implemented:** the prediction-usefulness evaluation from the presentation
   (forecasting with vs without analogies, Brier score on unresolved questions).
9. Human evaluation from the paper is not replicated.
10. **Two agentic defects were found and fixed on 2026-08-26; a third is still open.**
    Fixed: (a) the pipeline wrote *descriptions* where the task wants Wikipedia *titles*, so
    8/20 answers were scored against a page they did not name — one of them the input
    event's own article, scoring 0.00; (b) no agent knew an analogy must be a *different*
    event, so the loop drifted toward restatement. See `agentic_pipeline/canonical_names.py`
    and PROJECT.md 5.6. Still open: the generator sometimes names things that are not
    events at all (it answered with an ethnic group and a political party before the fix).
11. **A Wikipedia lookup that FAILS must never be cached (fixed 2026-08-23).** The provider
    used to store the empty result of a timed-out or throttled call, so one network blip
    marked an event permanently nonexistent. It invalidated a whole 10-hour popular run:
    1048 of 1171 cache entries became stored failures, and scored-example counts fell in
    *run order* rather than by method — agentic ran last and lost 14 of 20. **Diagnostic:
    if `n_evaluated` declines with run order, suspect caching or resource exhaustion, never
    the methods.** Now only answers are cached; failures are counted in `failed_calls` and
    warned about once.
12. **Never encode the metric into the method.** The distinct-event rule is phrased as "must
    be a distinct event" — the task definition, which the paper's own judging prompt states
    ("[Self-analogy is bad!!!]"). Expressing it as a Jaccard threshold at alpha=0.35 would be
    fitting the pipeline to its own scorer, and is not acceptable however much it scores.

---

## 9. Moving the project / switching to local LLMs

### What git carries, and what it does not

Carried: all code, `hal/project_defaults.py`, datasets, tests, docs, and the committed
results. **Not carried** (git-ignored): `.env` and `.hal_cache/`.

`.hal_cache/` (~55 MB on the old PC) is **deliberately not transferred** — decided
2026-08-21. It is regenerable, and switching models invalidates most of it anyway:

- `evaluation_mds.json` — keyed by evaluation model; a new judge must redo these.
- `embeddings/` — keyed by provider+model+dimension; only reusable while
  `EMBEDDING_MODEL` stays `gemini-embedding-001`.
- `wikipedia_en.json` — model-independent, but free to rebuild (no quota, just time).

Worst case on a fresh machine is re-embedding the 658 pool events (~658 calls, a separate
quota bucket from generation) the first time a retrieval baseline runs. Accept it.

Setup on the new machine:

1. `py -m pip install -r requirements_project.txt`
2. `copy .env.example .env` and add `GEMINI_API_KEY` (needed only by the two retrieval
   baselines, which use Gemini embeddings; everything else is local)
3. `ollama pull qwen3:8b` — and check `ollama --version` is **>= 0.9.0**. Older servers
   accept the `think` parameter and silently ignore it, so `LOCAL_THINK=false` becomes a
   no-op, qwen3 prepends a `<think>` block to every plain-text reply, and the MDS judge's
   score parser reads a digit out of the reasoning instead of the verdict. The laptop
   shipped with 0.6.8 and produced a 1 where the answer was 4.
4. `py -m pytest tests -q` — 270 tests, no key and no network required
5. **Check the tokenizer** (see below)

### The tokenizer decides part of the metric — check it on any new machine

MDS's literal-similarity term is NLTK tokenization + stop-word removal + Jaccard. NLTK's
corpora are *data*, not a package, so `pip` cannot install them. `hal/text_similarity.py`
therefore downloads `punkt`, `punkt_tab` and `stopwords` **automatically on first use** —
there is no manual step, and the `nltk.download(...)` command that used to be in
`requirements_project.txt` is obsolete.

If that download fails (offline machine, blocked network) the code silently falls back to a
regex tokenizer, which produces **slightly different Jaccard values and therefore different
MDS**. Measured on 60 real dimension pairs from our own results: only 17/60 identical, mean
difference 0.0067, max 0.0505. Small, but enough that results from the two tokenizers are
not strictly comparable.

Verify after setup:

```bash
py -c "from hal.text_similarity import tokenizer_backend; print(tokenizer_backend())"
```

It must print `nltk`. If it prints `regex`, get the machine online and re-run before
producing anything you intend to keep.

This is auditable after the fact too: every result row records `"tokenizer"`, and the
tokenizer is part of the condition key, so pooling rows scored by different tokenizers
triggers the same `MORE THAN ONE condition` warning as mixing models or settings.

### Running the LLM locally — IMPLEMENTED

`hal/providers/local.py` is built and tested. To use it:

```bash
ollama serve
ollama pull qwen3:8b
```
then in `.env`:
```
LLM_PROVIDER=local
LLM_MODEL=qwen3:8b
```
Nothing else changes — all seven methods and the evaluator go through
`get_llm(role=…)`. Verified end-to-end offline: the full agentic pipeline
(generate → critic → anti-analogy → judge → summarizer) runs against a stubbed
Ollama, and every request carries `num_ctx` and JSON mode.

Chosen model (2026-08-22): **qwen3:8b for every role, including the judge** — one model
keeps the condition clean. Hardware is an RTX 5060 Laptop with **8 GB VRAM**, which caps
usable models at the 7-9B class at Q4; 12B+ spills to RAM and crawls.

*Why not `llama3.1:8b`, the model the paper used?* Considered and rejected. The paper's
released code has **no Llama support at all** (`--model` accepts only chatgpt/gpt4/gemini)
and §5.1 never says how Llama3.1-8B was served, so there is no setup to faithfully match —
and Ollama's Q4 build would differ from their presumably full-precision one regardless.
Against that, qwen3 measured better here (MDS 4.002 vs 3.824, Pass@1 0.30 vs 0.10 on the
agentic method, n=20). The cost of the choice: the paper's published Llama3.1-8B table
(popular: Direct Gen. 3.90, Self-reflection 4.13) is no longer a sanity anchor for our
baseline reproductions.

Settings that matter, all tracked in `hal/project_defaults.py`:

* `LOCAL_NUM_CTX=10240` — **the critical one, and it is measured, not guessed.**
  A full-settings agentic example on `qwen3:8b` (140 calls) produced prompts of
  mean 1265 / p90 1473 / **max 6572** tokens. Ollama's default is 2k-4k and truncates
  *silently*: it drops the START of the prompt and answers anyway, so the model loses the
  task and the input event while still seeing the trailing "reply with JSON" instruction —
  producing well-formed JSON about the wrong thing. `hal/providers/local.py` now checks
  `prompt_eval_count` after every call and raises rather than let that pass.
  Context costs VRAM; measured on an 8 GB card with `qwen3:8b` Q4:
  `8192` → 100% on GPU, `10240` → 92%, `16384` → 80% (~45% slower per call).
* `LOCAL_THINK=false` — Qwen3 reasons before answering. Measured: 20-24 s with thinking
  vs **6.4 s** without, same answer. Off by default; flip to `true` to test whether
  thinking improves analogy quality (a cheap ablation, not an assumption).
* `LOCAL_API_STYLE=ollama` — the native `/api/chat`, the only style that accepts
  `num_ctx`. `openai` works for LM Studio / vLLM / llama.cpp but the window must then be
  set on the server.
* `format: "json"` constrains decoding so invalid JSON is impossible, and
  `LOCAL_KEEP_ALIVE=30m` keeps the model resident across the many calls per example.

**Verified against real Ollama (2026-08-21, desktop, RTX 2070 SUPER 8 GB):**
`llama3.1:8b` completed a smoke run (225 s), and **`qwen3:8b` completed a FULL-settings
example (2 rounds, 8 candidates, 4 ReAct steps) with zero errors** — 140 LLM calls,
3258 s, answer "Iranian Revolution". An 8B local model completed what Gemma 4 31B could
not, confirming Gemma's failures were format compliance rather than capability.

**Always run the agentic pipeline at FULL settings — decided 2026-08-22.**
`REFINEMENT_ROUNDS=2`, `MAX_CANDIDATES=8`, `REACT_MAX_STEPS=4`, and every candidate
reviewed (no `--critique-top-n` cap). These are the tracked defaults, so the plain command
already does the right thing; pass nothing extra. Reduced settings are the *method changed*,
not the method run cheaply, and results from them are not comparable with results from the
designed configuration. The only sanctioned exception is `--smoke`, which is explicitly a
plumbing check and writes to separate `*_smoke` files.

**Runtime planning — the call count is the cost, not the context.**
Full settings measured at **140 calls ≈ 54 min per example** (~23 s/call, on the old 16384
window that spilled 20% to CPU; 10240 with `LOCAL_THINK=false` should be faster).
Extrapolated: popular (20) ≈ 18 h, general (160) ≈ 6 days. There is no quota — only wall
time — so run overnight, use `--resume` freely, and consider `--limit` to build up the
general set in sittings rather than shrinking the method.

**Embeddings stay on `gemini-embedding-001` — decided 2026-08-21.** `EMBEDDING_MODEL` is
independent of `LLM_MODEL`, so going local for generation does **not** require changing
embeddings, and only two of the seven methods use them at all (`direct_retrieval`,
`twostage_retrieval`); the agentic pipeline never does. Cost is ~838 one-time calls
(658 pool + one per input event), then cached forever, on a quota separate from
`generate_content`.

Two alternatives were considered and rejected:

* *The paper's own vectors.* `dataset/similarity_embeddings-example.jsonl` holds the
  authors' OpenAI `text-embedding-3-small` vectors for all 658 pool events. All 160
  **general** input events are themselves in the pool, and 158 have byte-identical
  descriptions, so paper-exact retrieval is possible offline for general — but only
  **5 of 20 popular** input events are covered, so popular would still need an OpenAI key.
  Rejected to keep ONE embedding space across both datasets; a seam between popular and
  general would make our own two result tables non-comparable.
* *A local embedding model* (nomic-embed-text, bge-m3). Rejected for now: it changes a
  third variable at the same time as the generator and judge, and re-embeds the pool.

If this is ever revisited: the cache is keyed by provider+model+dimension, so a new
embedding model silently re-embeds all 658 events and makes retrieval numbers
non-comparable with existing results. Treat it as a new condition, new results folder.

**Research caution:** local-model results are a *different experimental condition*. Do not
merge them into the existing Gemma tables — start a new `results/automatic_evaluation_<model>/`
directory. Every result row already records `generation_model` and `evaluation_model`.

**Suggested first steps on the laptop, in order:**
```bash
py -m pytest tests -q                                   # everything offline must pass
py -c "from hal.text_similarity import tokenizer_backend; print(tokenizer_backend())"
                                                        # MUST print nltk (see above)
py examples/run_all_methods.py --dry-run                # no key, no network
py examples/run_all_methods.py --methods agentic --smoke --dataset popular --index 0
```

Also worth doing before any multi-hour job: re-run the VRAM probe on the new GPU. The
laptop's RTX 5060 idles nearly empty where the desktop had ~700 MB in use, so it may hold
more than `LOCAL_NUM_CTX=10240` fully on the GPU — free speed. Measure with a short call at
each candidate window and read `size_vram / size` from `http://localhost:11434/api/ps`;
anything below 100% is spilling to CPU.

---

## 10. Conventions

- Tests never hit the network; they use `hal/providers/mock.py` and ignore the local `.env`.
- Tests read the model from `PROJECT_DEFAULTS`, so changing the model must not require
  editing tests.
- Every LLM step requests JSON and validates it; malformed output degrades to a documented
  fallback and is recorded in `result.errors` — it never crashes a run.
- Logs record observable things only (query, tool, results, one-sentence rationale capped
  at 240 chars). Never store or print long chain-of-thought.
- Result rows carry reproducibility metadata: models, timestamp, α, dimension weights, mode.
- Caches are an optimisation only; a cache failure warns and continues, never aborts.
