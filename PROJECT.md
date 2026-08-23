# Past Meets Present with AI Agents

**An agentic extension of *"Past Meets Present: Creating Historical Analogy with Large Language Models"***

Daniel Ruderman, Yuval Rom

---

This repository is a fork of the code accompanying Li et al., *Past Meets Present: Creating
Historical Analogy with Large Language Models*. The original implementation is preserved
untouched under `framework/` and `evaluation.py` and remains the reference for the paper's
methods. Everything described below is **added** alongside it.

**Research question:** *Will transitioning to agentic AI-based analogy retrieval significantly
improve results over standard LLM prompting?*

---

## 1. Motivation

Applied history uses past events, patterns and precedents to reason about present-day problems.
Its central tool is the **historical analogy**: comparing a current event to a similar past one
in order to understand its causes, dynamics and likely consequences.

Analogies are powerful precisely because they shape how people interpret unfamiliar events —
and that is also why they are dangerous. Research in applied history shows that people, and
politicians in particular, tend to pick analogies that are:

- **superficial** — based on obvious surface similarities;
- **politically convenient** — chosen to support a position already held;
- **overused** — whichever famous example comes to mind first;
- **incomplete** — ignoring the differences that actually matter.

An LLM asked for a historical analogy exhibits the same failure modes, amplified by
pre-training biases: it gravitates to the canonical answer, it can hallucinate events that
never happened, and it never looks for evidence that its own answer is wrong.

Our hypothesis is that these are *search and criticism* failures rather than *knowledge*
failures, and that they are best addressed with an agentic architecture that explicitly
searches, criticises and challenges its own candidates before committing to one.

## 2. What historical analogy generation is

**Task (from the paper):** given an input event `E_I` and its description `D_I`, produce a
historical event `E_H` that is analogous to it.

A good analogy matches the **deep structure** of the events, not their wording. The paper
represents an event along four dimensions, and our project keeps that representation:

| Dimension | Question |
|---|---|
| **Topic** | What kind of event is it? |
| **Background** | What conditions and causes produced it? |
| **Process** | How did it unfold, through which mechanisms and actors? |
| **Result** | What were the consequences? |

Canonical example: COVID-19 ↔ the 1918 Spanish flu — both global outbreaks of respiratory
disease, both causing social and economic disruption, both forcing governments to act under
uncertainty.

**Datasets** (shipped with the original repository, in `dataset/`):

| File | Rows | Schema | Role |
|---|---|---|---|
| `popular_analogy.jsonl` | 20 | `event_name`, `event_intro`, `target_event` | famous analogies with a reference answer → Pass@1 |
| `general_analogy.jsonl` | 160 | `event_name`, `event_intro`, `event_type` | harder cases with no single correct answer → MDS. Themes: War (50), Politics (50), Culture and Society (50), Economy (10) |
| `event_pool.jsonl` | 658 | `url`, `history_event_text`, `history_time_text`, `history_intro_text` | the Google Arts & Culture event pool searched by the retrieval methods |
| `similarity_embeddings-example.jsonl` | 658 | `url`, `embeddings` | the authors' pre-computed OpenAI `text-embedding-3-small` vectors (1536-d, unit norm) |

## 3. What the original paper did

Six methods in two families (`framework/`):

**Dataset retrieval** — search a fixed event pool.

1. **Direct Retrieval** — embed the input description and every pool description, return the
   event with the highest cosine similarity.
2. **Two-stage Retrieval** — retrieve the top-10 by cosine similarity, then let an LLM choose
   the best analogy from that candidate set.

**Free generation** — use the knowledge inside the model.

3. **Direct Generation** — one prompt, one analogous event. Simple, but prone to hallucination
   and to stereotyped, surface-level answers.
4. **Two-stage Generation** — generate **10** candidates, verify each one through Wikipedia
   (dropping events with no entry), then select the best.
5. **Generation with Summarizing** — summarise the input event *and* every candidate into the
   four dimensions, and compare the structured summaries instead of raw descriptions.
6. **Self-reflection Framework** — the paper's most agent-like method. A **Candidate
   Generator** proposes **5** candidates; an **Answer Reflector** either emits a final answer
   or a `Reflection` telling the generator how to change the set; the loop repeats. Candidates
   are Wikipedia-verified. This is the paper's strongest method.

**Evaluation** (`evaluation.py`):

- **Pass@1** for popular analogies (Wikipedia title sets of answer and reference must intersect).
- **MDS (Multi-Dimensional Similarity)** for general analogies:

  ```
  MDS = Σ_d  w_d · sim_Abs(D_I^d, D_H^d) · max(α − sim_Lit(D_I^d, D_H^d), 0)
  ```

  where `d ∈ {topic, background, process, result}`, `sim_Abs` is an LLM judgement on a 1–4
  scale, `sim_Lit` is Jaccard similarity over stop-word-filtered NLTK tokens, the weights are
  `w = {topic 0.5, background 1, process 2, result 2}` and `α = 0.35`. High abstract similarity
  is rewarded; high *literal* similarity is penalised, because an analogy that works only
  because both events share names, countries or keywords is not a good analogy. The metric
  correlates with human ranking at Kappa 0.67 / Pearson 0.72 / Spearman 0.73.

## 4. Is the paper's self-reflection framework already agentic?

Partially. It splits the task into two roles, it revises candidates from feedback, and it uses
Wikipedia to reduce hallucination. But it is **not** a ReAct-style agent:

| ReAct | The paper's self-reflection |
|---|---|
| interleaves reasoning, actions and observations | fixed workflow |
| the model decides which tool to use next | Wikipedia is used only for verification |
| observations update the next reasoning step | reflection is over candidate quality alone |
| open-ended investigation | no open-ended search |

Our project extends it into a fully agentic pipeline with search, criticism, anti-analogies and
a final judgement.

## 5. Our agentic pipeline

```
  Input event / analogy prompt
             │
             ▼
   ┌───────────────────────┐        critique feedback        ┌──────────────────┐
   │  Generate/Search      │ ──────────────────────────────► │  Critic agent    │
   │  agent                │ ◄────────────────────────────── │  evaluates each  │
   │  proposes 5–10        │                                 │  candidate       │
   │  candidate analogies  │        counterexample feedback   └──────────────────┘
   │                       │ ──────────────────────────────► ┌──────────────────┐
   │                       │ ◄────────────────────────────── │ Anti-Analogy     │
   └───────────────────────┘                                 │ agent finds      │
             │        └──── iterative refinement loop, 1–3 ──►│ counterexamples │
             │                                                └──────────────────┘
             ▼   refined candidates + critiques + counterexamples + evidence
      ┌──────────────┐        ┌────────────────────┐
      │ Final Judge  │ ─────► │ Final Summarizer   │
      │ ranks them   │        │ explains the winner│
      └──────────────┘        └────────────────────┘
```

Three components are **agents** (they run a ReAct-style tool loop): Generate/Search, Critic and
Anti-Analogy. **Final Judge is not an agent** — it is a ranking/evaluation component that runs
after the loop, with no tools and no ReAct loop. Final Summarizer is the final explanation
stage, a plain LLM call.

### 5.1 Generate/Search agent — `agentic_pipeline/generate_search_agent.py`

Takes the input event, understands its historical structure and proposes 5–10 candidate
analogies (`MAX_CANDIDATES`, default 8). It is instructed to prefer structural analogy over
surface word similarity and to vary period, region and causal pattern rather than produce
variations of one idea. It can search before committing to a candidate, and every candidate is
checked against the knowledge base — unverifiable events are **flagged** (`verified=False`)
rather than silently dropped, so the Critic can comment on them and the Judge can penalise them.

In later rounds it receives the candidates, the critiques, the counterexamples and the
evidence, and returns a revised set in which each candidate is explicitly **kept**, **revised**,
**replaced**, **added** or dropped, with a reason. The prompt states that a candidate should not
be replaced merely for having been criticised: a strong candidate with known limitations is
more useful than a weak one nobody attacked.

### 5.2 Critic agent — `agentic_pipeline/critic_agent.py`

Evaluates **each** candidate and returns a structured `Critique`:

- 1–4 scores for **topic / background / process / result** similarity (the paper's scale);
- the **structural correspondence** that genuinely holds;
- **important differences** (scale, period, technology, institutions);
- **weak assumptions** the analogy depends on;
- **factual/evidence problems**;
- whether the analogy is **mostly surface-level**;
- an overall score and a `keep` / `revise` / `replace` recommendation.

It may search before criticising a factual claim. The output is structured precisely so the
Generate/Search agent can act on it rather than re-reading prose.

### 5.3 Anti-Analogy agent — `agentic_pipeline/anti_analogy_agent.py`

The component that keeps the system honest. For each candidate it actively looks for:

- historically similar cases that produced **different or opposite outcomes**;
- counterexamples showing the mechanism does not reliably produce the claimed result;
- cases sharing the apparent pattern but diverging at a decisive point;
- evidence that reasoning from this analogy has misled before.

It returns counterexamples (each verified against the knowledge base where possible), failure
modes, a **robustness** score in [0, 1] and a verdict `holds` / `weakened` / `undermined`. This
lets the Generate/Search agent weaken confidence in a candidate, narrow it, replace it, or
record its limitations — instead of accumulating support for whatever it thought of first.

### 5.4 ReAct-style search — `agentic_pipeline/react.py`

A small loop implemented directly (no agent framework), so every step is visible and
modifiable. Each turn the agent returns one JSON object: either a `search`/`lookup` action, or
`finish` with the answer. Observations are appended and the loop continues until the agent
finishes or exhausts its tool budget (`REACT_MAX_STEPS`).

Search goes through a `SearchProvider` abstraction. The default is
`WikipediaSearchProvider` — free, and the source the paper already uses. A web-search backend
can be registered later without touching any agent.

**Logging policy:** logs record *observable* information — the search query, the tool used, the
titles returned, the candidate, a one-sentence rationale (capped at 240 characters), the
critique, the counterexample and the revision. Long internal monologue is never requested,
stored or printed.

### 5.5 Iterative refinement loop — `agentic_pipeline/pipeline.py`

```
candidates ← Generate/Search.propose(event)

for round in 1..REFINEMENT_ROUNDS:            # default 2, design range 1–3
    critiques      ← Critic.critique_all(candidates)
    anti_analogies ← AntiAnalogy.investigate_all(candidates)
    candidates, revisions ← Generate/Search.revise(candidates, critiques,
                                                   anti_analogies, evidence)

if the final set changed in the last round:   # otherwise reuse the feedback
    critiques, anti_analogies ← review(candidates)

ranking ← FinalJudge.rank(candidates, critiques, anti_analogies, evidence)
result  ← FinalSummarizer.explain(ranking.winner)
```

Every round is recorded as a `RefinementRound`: critiques, counterexamples, revisions and the
candidate set after the round.

### 5.6 Final Judge — `agentic_pipeline/final_judge.py`

**Not an agent.** No tools, no ReAct loop, no ability to propose new candidates — it is a
ranking/evaluation stage after the loop. It receives the refined candidates, the Critic
feedback, the Anti-Analogy counterexamples and the collected evidence, and ranks all candidates
on structural quality, evidence, major differences, unresolved critiques, counterexamples and
robustness. Each row of the ranking carries a rank, the candidate event, a concise reason and
its important weaknesses.

If the judge's output cannot be parsed, `heuristic_ranking()` produces a deterministic ranking
from the structured scores (`critic_overall × robustness`, penalised for unverified and
surface-level candidates), so a run always yields a result. That function is also available on
its own for an ablation: *how much does the LLM judge add over the structured scores?*

### 5.7 Final Summarizer — `agentic_pipeline/final_summarizer.py`

A plain LLM call that explains the winning analogy to someone who wants to *use* it: the input
event, the winning analogy, what the comparison illuminates, the important structural
similarities and differences, the relevant counterexamples, the limitations, and why it ranked
above the alternatives.

### 5.8 Structured data — `hal/schemas.py`

Components exchange dataclasses, not free-form strings: `HistoricalEvent`, `EventDimensions`,
`CandidateAnalogy`, `Critique`, `CounterExample`, `AntiAnalogyReport`, `CandidateRevision`,
`RefinementRound`, `RankedCandidate`, `JudgeRanking`, `Evidence`, `FinalAnalogyResult`. Every
LLM step requests JSON, and `hal/json_utils.py` validates it tolerantly (markdown fences,
surrounding prose, Python literals, trailing commas). Malformed output never aborts a run:
each component degrades to a documented fallback and records the problem in `result.errors`.

`FinalAnalogyResult.to_output_row()` emits the original repository's jsonl format
(`event_name`, `event_intro`, `analogy_event`, `candidate`), so our results feed straight into
the paper's evaluation.

## 6. Model / provider abstraction

The provider is **orthogonal to the method**, so the same algorithm can be run against
different models:

```
LLMProvider.generate(prompt, stop=…, temperature=…, json_output=…)   hal/providers/base.py
EmbeddingProvider.embed(text) / embed_batch(texts)
SearchProvider.search(query, top_k) / get_page(title)
```

| Concrete implementation | File |
|---|---|
| `GeminiLLMProvider`, `GeminiEmbeddingProvider` | `hal/providers/gemini.py` |
| `WikipediaSearchProvider`, `NullSearchProvider` | `hal/providers/search.py` |
| `MockLLMProvider`, `MockEmbeddingProvider`, `MockSearchProvider` | `hal/providers/mock.py` |
| `CachedEmbeddingProvider` (decorator) | `hal/providers/caching.py` |

Algorithms only ever call the factory:

```python
from hal.providers import get_llm, get_embedding_provider, get_search_provider

llm = get_llm(role="critic")          # model comes from CRITIC_MODEL or LLM_MODEL
```

**Adding a provider** (e.g. OpenAI, a local model) means writing one subclass and registering
it — no method code changes:

```python
from hal.providers import register_llm
register_llm("my_provider", lambda settings, role, model: MyLLM(model=model))
# then: LLM_PROVIDER=my_provider
```

**Gemini specifics.** We use the current official SDK `google-genai` (`from google import
genai`) and the standard `GEMINI_API_KEY` variable. The Gemini API exposes two generation
surfaces — the long-standing `client.models.generate_content` and the newer
`client.interactions.create`; `GEMINI_API_SURFACE` selects one, and the default `auto` prefers
`models.generate_content` (it supports stop sequences and JSON mime types directly) and falls
back to `interactions.create`. All of this lives inside the provider: the API adaptation is
separated from the research methodology. Safety filters are set to `BLOCK_NONE`, matching the
original repository, because historical content about wars and atrocities otherwise gets
blocked.

### Configuration: shared defaults vs. local secrets

Configuration is split in two, so that a team of collaborators shares research settings through
Git while keeping credentials private:

| | file | tracked? | contains |
|---|---|---|---|
| **Shared project defaults** | `hal/project_defaults.py` | **yes** | provider, models, refinement rounds, candidate counts, temperatures, retry/cache settings |
| **Local secrets & overrides** | `.env` | **no** (git-ignored) | `GEMINI_API_KEY`, plus any personal override |

Values resolve in this order:

1. **environment variable** (including anything loaded from your local `.env`) — always wins;
2. **`hal/project_defaults.py`** — the tracked, shared project default;
3. a code-level fallback in `hal/config.py` — only reached if a key is removed from the tracked
   defaults.

So editing `hal/project_defaults.py` and pushing changes the default *for the whole team*, while
setting an environment variable changes it *only for you*, without touching tracked files:

```bash
LLM_MODEL=some-other-model python examples/run_all_methods.py --methods agentic
```

**Secrets never get a tracked default.** `GEMINI_API_KEY` (and the other keys in `SECRET_KEYS`)
are refused by `hal/project_defaults.py`, which raises at import time if one is ever added
there. No API key appears in any Python file, tracked config, `.env.example`, documentation or
test.

`hal.config.config_source("LLM_MODEL")` reports where a value came from (`env`, `project` or
`fallback`) if you need to check what is actually in effect.

| Variable | Tracked default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | *(none — secret, local `.env` only)* | your key |
| `LLM_PROVIDER` / `LLM_MODEL` | `local` / `qwen3:8b` | LLM for every role (local Ollama server) |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | `gemini` / `gemini-embedding-001` | retrieval embeddings |
| `EMBEDDING_DIMENSIONS` | *(SDK default)* | optional reduced dimensionality |
| `SEARCH_PROVIDER` | `wikipedia` | agent tool backend |
| `GENERATOR_MODEL`, `CRITIC_MODEL`, `ANTI_ANALOGY_MODEL`, `JUDGE_MODEL`, `SUMMARIZER_MODEL`, `BASELINE_MODEL`, `EVALUATION_MODEL` | = `LLM_MODEL` | optional per-role models |
| `REFINEMENT_ROUNDS` | `2` | refinement rounds (design range 1–3) |
| `MAX_CANDIDATES` / `MIN_CANDIDATES` | `8` / `5` | candidate-set size |
| `REACT_MAX_STEPS` / `SEARCH_TOP_K` | `4` / `4` | tool budget per agent call |
| `LLM_TEMPERATURE` | `0.1` | the paper's baseline temperature |
| `EVALUATION_TEMPERATURE` | `0.0` | MDS judge |
| `MAX_RETRIES`, `RETRY_BASE_DELAY`, `RETRY_MAX_DELAY`, `REQUEST_DELAY` | `5`, `2.0`, `60.0`, `0.0` | quota robustness |
| `CACHE_ENABLED` / `CACHE_DIR` | `true` / `.hal_cache` | on-disk caches |
| `GEMINI_API_SURFACE` | `auto` | `auto` / `models` / `interactions` |

> **Model names are configuration, not architecture.** `qwen3:8b` is the current team
> default and it is written in exactly one place (`hal/project_defaults.py`); no algorithm file
> mentions a model name. Gemini model availability and quotas change over time — check your
> limits in AI Studio, and change the default for everyone by editing that one file and pushing.
> The generation model and the embedding model are independent: switching `LLM_MODEL` does not
> touch `EMBEDDING_MODEL`.

### Comparing methods across models

Because the provider is orthogonal to the method, the planned experiment grid works by changing
environment variables only — no tracked file and no algorithm changes:

```bash
python examples/run_all_methods.py --methods direct_generation                      # qwen3:8b (tracked default)
LLM_MODEL=qwen3:8b python examples/run_all_methods.py --methods direct_generation
LLM_MODEL=qwen3:8b python examples/run_all_methods.py --methods agentic
```

The runner also accepts `--model NAME` / `--provider NAME`, which override the configuration for
that single run.

## 7. Quota, caching and robustness

- **Retries with exponential backoff + jitter** on every outbound call (`hal/retry.py`).
  Transient failures (429, 5xx, timeouts) are retried and a server-provided `retryDelay` is
  honoured; permanent failures (bad key, unknown model) fail fast instead of burning quota.
- **Embedding cache** (`hal/cache.py`) keyed by *provider + model + dimensionality + text hash*,
  so vectors from different embedding models never mix. The 658-event pool is embedded once.
- **Wikipedia cache** for page lookups and searches. It stores only *answers*, never
  *failures* — see below.
- `REQUEST_DELAY` adds a fixed pause before each call if you hit per-minute limits.
  `WIKI_REQUEST_DELAY` (default 0.2 s) does the same for MediaWiki alone, since a local
  Ollama needs no pacing but Wikipedia does.
- Caches only avoid repeating identical external calls; they are never part of a result.

### A cached failure is not a result (bug found 2026-08-22, run invalidated)

`WikipediaSearchProvider` used to write an empty result to the cache whenever a lookup threw:

```python
except Exception:
    titles = []              # a timeout, a 429, a reset connection
self._cache.set(key, titles) # ...remembered forever as "no such event"
```

The first full `popular` run on `qwen3:8b` issued thousands of unpaced MediaWiki calls over
about ten hours. Wikipedia throttled it hard enough that lookups failed even after three
retries each, and every failure was then cached permanently. **1048 of the 1171 cache entries
(89%) were stored failures**, including real articles such as `Vietnam War`.

The damage was invisible and progressive — each method inherited a more poisoned cache than
the last, so the results degraded in run order rather than by method quality:

| method | started | scored | discarded |
|---|---|---|---|
| direct_retrieval | 10:37 | 25 | 0 |
| twostage_retrieval | 16:36 | 14 | 6 |
| direct_generation | 16:37 | 17 | 4 |
| twostage_generation | 17:30 | 7 | 13 |
| summary_generation | 18:41 | 7 | 13 |
| reflection_generation | 19:36 | 4 | 16 |
| agentic | 21:01 | 6 | 14 |

The agentic pipeline ran last and lost the most: its MDS of 3.173 rested on 6 of 20 examples.
Re-resolving its answers against a clean cache, **all 20 resolve** — nothing was wrong with
the answers. That run is void and must not be quoted.

Now: a failed *call* returns empty without being cached, while an empty answer from a call
that *succeeded* (Wikipedia genuinely reporting no such page) is still cached. The provider
also counts failures in `failed_calls` and prints a warning the first time one occurs, because
a run that silently loses thousands of lookups is indistinguishable from a model inventing
thousands of fake events. `tests/test_wikipedia_cache.py` pins both halves.

## 8. Installation

### Setup after cloning or pulling (this is the whole checklist)

1. **Install dependencies** — `pip install -r requirements_project.txt`
2. **Create `.env`** — `cp .env.example .env`
3. **Add your own `GEMINI_API_KEY`** to it

That is all. Everything else — the LLM provider, `qwen3:8b`, the embedding model, the
refinement rounds, the candidate counts, the temperatures — already comes from the tracked
repository (`hal/project_defaults.py`), so both collaborators run the same configuration without
coordinating anything by hand.

> **Why `.env` is not committed:** it holds your personal API key and any local override. It is
> listed in `.gitignore` on purpose, so it never reaches the shared history — which also means a
> project default placed only in `.env` would *not* reach your partner. Shared settings belong in
> `hal/project_defaults.py`; that file is committed, so `git push` / `git pull` propagates them.

### Details

```bash
pip install -r requirements_project.txt
```

Optional, for the faithful NLTK tokenizer used by the literal-similarity metric:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

(Without NLTK a regex tokenizer is used; `hal.text_similarity.tokenizer_backend()` reports which
one is active.)

Then configure your key:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

and put your key in `.env`:

```
GEMINI_API_KEY=your-key-here
```

`.env` is git-ignored. Alternatively export `GEMINI_API_KEY` in your shell. Get a key at
<https://aistudio.google.com/apikey>. Each developer uses their own key.

You do **not** need to copy the other values from `.env.example` into `.env` — they are all
commented out there and documented as optional. Uncomment one only when you want to override the
shared default on your machine.

## 9. Running

### Quick smoke test (few API calls)

```bash
python examples/run_all_methods.py --smoke
```

Small event pool (40), one refinement round, three candidates, one tool call per agent step.

Completely offline, no key and no network — checks the plumbing only:

```bash
python examples/run_all_methods.py --dry-run
```

### All methods on a real example

```bash
python examples/run_all_methods.py --dataset popular --index 0
```

The first run embeds all 658 pool events (cached afterwards). Useful options:
`--dataset popular|general`, `--index N`, `--methods all|baselines|agentic|<comma list>`,
`--rounds N`, `--max-candidates N`, `--react-steps N`, `--critique-top-n N`, `--model NAME`,
`--provider NAME`, `--pool-limit N`, `--full`, `--output results.jsonl`, `--verbose`.

### Only our agentic pipeline

```bash
python examples/run_all_methods.py --methods agentic --dataset popular --index 0
```

or, over a whole dataset:

```bash
python -m agentic_pipeline.pipeline --dataset popular --rounds 2 --output agentic_output.jsonl
```

### Individual baselines

Same interface as the original scripts:

```bash
python -m gemini_baselines.reflection_generation --testset popular --output output.jsonl
python -m gemini_baselines.direct_generation     --testset general --limit 5
python -m gemini_baselines.twostage_retrieval    --testset popular --pool-limit 100
```

`--testset` accepts `popular`, `general` or a path to any `.jsonl` whose rows have
`event_name` and `event_intro`.

### Evaluation

```bash
python -m gemini_baselines.evaluation_mds --testset output.jsonl
python -m gemini_baselines.evaluation_mds --testset output.jsonl --pass1   # popular set
```

Input is the output format of any method (`event_name`, `event_intro`, `analogy_event`).

### Tests

```bash
python -m pytest tests -q
```

100 tests, no API key and no network required — everything runs against mock providers.

## 10. What we added

```
PROJECT.md                     this document (research/technical detail)
PROJECT_GUIDE.txt              beginner-friendly practical guide: folders, configuration,
                               API key setup, how to run things, troubleshooting
.env.example                   local-configuration template: the required secret plus
                               commented-out optional overrides (never contains a key)
requirements_project.txt       our dependencies (the paper's are untouched)
.gitignore                     ignores .env, caches and run outputs

hal/                           shared infrastructure (no research method here)
  project_defaults.py          TRACKED shared project defaults (model, rounds, …) — edit
                               this to change a default for the whole team
  config.py                    resolution order env > project defaults > fallback,
                               per-role models
  schemas.py                   HistoricalEvent, CandidateAnalogy, Critique, CounterExample,
                               CandidateRevision, JudgeRanking, FinalAnalogyResult, …
  providers/
    base.py                    LLMProvider / EmbeddingProvider / SearchProvider interfaces
    gemini.py                  GeminiLLMProvider, GeminiEmbeddingProvider
    search.py                  WikipediaSearchProvider, NullSearchProvider
    mock.py                    fake providers for tests and --dry-run
    caching.py                 CachedEmbeddingProvider
    factory.py                 registry: get_llm(role=…), get_embedding_provider(), …
  cache.py                     JsonCache, EmbeddingCache (keyed by model)
  retry.py                     exponential backoff, transient/permanent classification
  wiki.py                      Wikipedia helper with the original's fallback semantics
  json_utils.py                tolerant parsing of LLM JSON
  text_similarity.py           Jaccard / literal similarity (algorithmic, no LLM)
  vector.py                    cosine similarity (numpy optional)
  io_utils.py                  jsonl helpers, dataset paths

gemini_baselines/              provider-neutral ports of the paper's six methods
  prompts.py                   every original prompt, copied verbatim
  common.py                    BaselineContext + shared steps
  direct_retrieval.py          ← framework/retrieval-based/direct_retrieval.py
  twostage_retrieval.py        ← framework/retrieval-based/twostage_retrieval.py
  direct_generation.py         ← framework/generation-based/direct_generation.py
  twostage_generation.py       ← framework/generation-based/twostage_generation.py
  summary_generation.py        ← framework/generation-based/summary_generation.py
  reflection_generation.py     ← framework/generation-based/reflection_generation.py
  evaluation_mds.py            ← evaluation.py  (Pass@1 + MDS)
  cli.py                       shared command-line plumbing

agentic_pipeline/              OUR new method
  prompts.py                   our prompts (Generate/Search, Critic, Anti-Analogy, Judge, Summarizer)
  react.py                     the ReAct-style tool loop
  generate_search_agent.py     Agent 1
  critic_agent.py              Agent 2
  anti_analogy_agent.py        Agent 3
  final_judge.py               Final Judge (NOT an agent) + heuristic_ranking
  final_summarizer.py          Final Summarizer
  pipeline.py                  the refinement loop + CLI

automatic_evaluation/          the automatic-evaluation runner (no method logic)
  methods.py                   maps a method name to its existing implementation
  runner.py                    generate / score / resume / aggregate / CSV

examples/run_all_methods.py    runs all six baselines + our pipeline on one dataset example
examples/run_automatic_evaluation.py
                               scores all seven methods with the paper's MDS + Pass@1
tests/                         157 tests, all offline
```

**Unchanged original files:** `framework/**`, `evaluation.py`, `dataset/**`, `README.md`,
`images/**`. Nothing in the original implementation was deleted, renamed or rewritten.

## 11. Research design constraints

**A. The baselines stay faithful.** `gemini_baselines/` reproduces the paper's methodology:
the same prompts (copied verbatim into `gemini_baselines/prompts.py`), the same candidate
counts (10 for two-stage/summarizing, 5 for self-reflection, top-10 for retrieval), the same
algorithm structure, the same Wikipedia verification, the same iteration logic, the same
temperature (0.1), the same input and output formats, the same evaluation. They contain **no**
Critic, **no** Anti-Analogy and no extra search — improving them would make the comparison
unfair.

**B. Our agentic method** is entirely in `agentic_pipeline/`.

**C. The model provider is orthogonal to both**, so `method × model` is a clean experiment grid.

### Provider differences (unavoidable, documented)

| Aspect | Original | Ours | Effect |
|---|---|---|---|
| Embeddings | OpenAI `text-embedding-3-small` (1536-d, unit-norm), pre-computed in `dataset/similarity_embeddings-example.jsonl` | configurable `EmbeddingProvider`, `gemini-embedding-001` by default, computed and cached | **The algorithm is identical** (cosine similarity over description embeddings; top-1 for direct retrieval, top-10 for two-stage). Vectors are *not* numerically comparable across embedding models, so retrieval numbers will differ from the paper's. Gemini vectors are L2-normalised so that an inner product equals cosine similarity, matching the original's use of `np.dot`. **Why not reuse the authors' shipped vectors?** They cover the 658 pool events, and every general input event is itself in the pool (158/160 with byte-identical text), so paper-exact retrieval on *general* would be possible with no API key — but only 5 of the 20 *popular* input events are covered. We use one embedding space for both datasets so that our popular and general retrieval results stay comparable to each other; reproducing the paper's exact retrieval on general remains an option for a future ablation. |
| LLM | `gpt-3.5-turbo` / `gpt-4` via LangChain, plus an old `google.generativeai` `gemini-pro` helper | any registered provider; Gemini via `google-genai` | Different model → different outputs. This is intended: the presentation's baseline is to re-run the paper's methods on current SOTA models. |
| Wikipedia | `wikipedia` PyPI package (HTML scraping, random pick on disambiguation) | MediaWiki API via `requests`, behind `SearchProvider`, cached | Same source and same procedure (exact title, else first search hit, lead section, truncated at 4096 chars). More reliable and reproducible; disambiguation resolution is deterministic rather than random. |
| Reflection loop | `while 'Reflection' in choice` with no bound | identical, plus a `max_reflections` cap (default 5) | The original cannot terminate if the model always reflects. The paper reports reflection firing in ~10% of cases, so the cap does not normally bind. |
| Conversation memory | LangChain `LLMChain` + `ConversationBufferMemory` | `ConversationMemory` reproducing the same buffer format (`Input: …` / `Output: …`) | The prompt the model sees is the same, without the LangChain dependency. |
| Malformed output | `ast.literal_eval` raises; a batch run dies | the original repair prompt is kept, then a local JSON/array extraction, then an empty result | The original's repair step is preserved; the extra fallbacks only prevent a crash. |
| MDS abstract score | out-of-range scores warn but are returned; unparseable output raises | clamped to [1, 4]; unparseable scores 1 | Keeps a batch run alive; identical for well-formed answers. |
| Literal similarity | NLTK tokenizer + stop-words | the same when NLTK is installed, else a regex tokenizer with NLTK's stop-word list | `tokenizer_backend()` reports which is active. Use NLTK for reported numbers. |
| `summary_generation.py` | never imports `llm_tools` → `NameError` when run | routes through the provider abstraction | A bug in the original; no prompt or step was changed. |
| `direct_retrieval.py` | reads `event_pool.jsonl.jsonl` and `similarity_embeddings.jsonl` (neither name exists in the repo) | reads `dataset/event_pool.jsonl` and computes vectors | Path bug in the original. |

## 12. Evaluation strategy

**First strategy — MDS.** We use the paper's metric to compare our agentic pipeline against the
baselines, with the same four dimensions, weights and threshold, and the LLM-judged part routed
through the configurable provider (`gemini_baselines/evaluation_mds.py`). Because MDS penalises
literal similarity, it will not reward an analogy that merely reuses the input event's
vocabulary. Note that MDS values are comparable only when the *same* judge model is used for
every method being compared.

**Second strategy — usefulness for prediction (planned, not implemented).** Test whether better
analogies improve forecasting: generate analogies with each method, ask the same LLM to predict
an outcome with and without them, and score with Brier/log score. To avoid data leakage from
resolved historical questions, this must use *unresolved* forecasting questions (ForecastBench,
Metaculus FutureEval), recording predictions before resolution. Not part of this codebase yet.

### 12.1 The automatic evaluation runner

`automatic_evaluation/` + `examples/run_automatic_evaluation.py` apply the paper's automatic
evaluation to all seven methods. The metric lives in `gemini_baselines/evaluation_mds.py` (our
port of `evaluation.py`); the runner only orchestrates.

**Faithfulness to `evaluation.py`.** Same four dimensions; same two-template
`extract_features` (the second template uses the already-summarised input event as the in-context
example); same 1–4 abstract-similarity prompt and integer parsing; same `jacc` (NLTK tokenize →
stop-word removal → Jaccard over token *sets*); same aggregation, including the fact that the
per-dimension `*All` value is `abstract × max(α − literal, 0)` **without** the weight, while the
weights enter only in the final sum:

```
MDS = 0.5·T_All + 1·B_All + 2·P_All + 2·R_All        α = 0.35
```

Pass@1 uses the original's rule — Wikipedia *search* both the reference and the produced name and
count a hit when the result sets intersect — not string equality. It is reported for the popular
set only; the general set has no reference answer.

**Separation of concerns.** The runner never re-implements a method: `automatic_evaluation/
methods.py` dispatches to the existing `gemini_baselines` functions and to
`AgenticAnalogyPipeline`. The agentic pipeline is scored on its **final winning analogy** only;
its ranking, round count and candidate history are stored as metadata and are excluded from the
metric, so all seven methods pass through an identical scoring path.

**Two independent models.** The model that *produces* an analogy (`LLM_MODEL` / `BASELINE_MODEL`)
and the model that *judges* it (`EVALUATION_MODEL`, inheriting `LLM_MODEL`) are configured
separately and both are recorded in every result row. MDS values are only comparable across
methods judged by the same model.

**Deviations, all infrastructural.** The judge is a configurable provider (Gemini by default)
rather than GPT-4 — abstract-similarity scores are therefore not numerically comparable with the
paper's tables, though the procedure is identical. Where `evaluation.py` raises inside `wiki()`
for an unresolvable event, we record a status (`no_analogy` / `unresolved_event`) and exclude the
sample from the averages instead of aborting the batch; such samples still count as Pass@1
misses, matching the original's division by `len(dataset)`. A failed API call is never given a
fabricated score.

**Reading the model's answer — a deviation forced by chat-tuned models (2026-08-22).**
The paper's prompts are *completion*-style: they demonstrate a layout, break off mid-pattern
(`Historical Analogies Events:`, or the clause TWOSTAGE_CHOICE trails off in), and rely on the
model continuing with the event name alone. GPT-4 does that. A chat-tuned local model instead
restarts the layout and puts its answer in the slot several lines down, in markdown. Two
consequences, both in *output handling* — **no prompt was changed**:

* `stop=["\n"]` in `direct_generation`, `twostage_generation` and `summary_generation` truncated
  the reply at its first line, which for such a model is `"==== Answer"`. The answer was never
  generated at all. The stop is dropped; the reply now runs to completion.
* `clean_answer` took line one. `extract_analogy_answer` (in `gemini_baselines/common.py`) now
  reads the name from the slot the prompt itself establishes, strips markdown, and returns `""`
  — an honest `no_analogy` — when the reply contains only scaffolding. For a bare event name,
  the format the paper's models produce, it returns exactly what `clean_answer` returned; the
  tests pin that equivalence.

Measured on 5 popular examples with `qwen3:8b`: before, 0/5 answers were usable and Pass@1 was
0.00; after, 5/5 were real event names and 2 were exact reference matches. Nothing was
laundered — two of the five are genuine failures where the model parroted `Spanish flu` from
the prompt's one-shot example, and they are still recorded as that answer. Dropping the stop
costs wall time (~30 s per direct-generation example instead of a truncated call).

This is deliberately a parsing change, not a prompt change. Rewording a prompt to demand a
terse answer would also change *which* event the model picks — the quantity being measured —
and no amount of testing could separate "the baseline got readable" from "the baseline got
better". Extraction runs after the model has already committed to an answer, so it cannot.

**Cost.** Judging one answer costs exactly 6 LLM calls (2 dimension summaries + 4 abstract
similarities). Results are cached in `.hal_cache/evaluation_mds.json`, keyed by the evaluation
model and a `PROMPT_VERSION` string, so the input event's summary is computed once and shared
across all seven methods and identical answers are judged once. The cache cannot change a score.

## 13. Status and next steps

Implemented and tested offline: the provider/search abstractions, all six provider-neutral
baselines, the MDS evaluation, the full agentic pipeline, the example runner and 100 tests.

Still to do:

- run the real Gemini smoke test once an API key is configured (see §9);
- confirm wall-time budget for the team default (`qwen3:8b`, local): there is no quota,
  but a full agentic run is measured in hours;
- full-dataset runs and MDS comparison of the baselines against our pipeline;
- the human-evaluation protocol from the paper, if we replicate it;
- the prediction-usefulness evaluation of §12;
- optionally, additional providers (open models such as Gemma / MiniMax, per the presentation's
  baseline slide) — a subclass plus a `register_llm` call each.
