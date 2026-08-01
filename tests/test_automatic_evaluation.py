"""The automatic evaluation runner (paper MDS + Pass@1), offline.

Every test uses mock providers: no API key, no network, no quota.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from automatic_evaluation import (
    ALL_METHODS,
    METHOD_LABELS,
    AutomaticEvaluation,
    EvaluationConfig,
    GenerationConfig,
    MethodRunner,
    QuotaExhausted,
    aggregate,
    canonical,
    expand_methods,
    is_quota_error,
    print_summary_table,
    write_summary_csv,
)
from automatic_evaluation.runner import (
    COMPONENT_COLUMNS,
    _detailed_path,
    _generations_path,
    _summary_path,
    load_completed,
)
from gemini_baselines.evaluation_mds import (
    DIMENSION_WEIGHTS,
    DIMENSIONS,
    LITERAL_THRESHOLD,
    EvaluationContext,
    abstract_similarity,
    component_scores,
    extract_features,
    jacc,
    mds_from_scores,
    pass_1_single,
    score_sample_detailed,
)
from hal.providers.mock import MockLLMProvider
from hal.retry import ProviderError
from hal.wiki import WikipediaHelper

FOUR_DIM = ("1. Summary: A wave of uprisings. 2. Background: Economic stagnation. "
            "3. Process: Protests spread. 4. Result: Some regimes fell.")
OTHER_DIM = ("1. Summary: A European revolutionary cascade. "
             "2. Background: Monarchical repression. 3. Process: Barricades rose. "
             "4. Result: Restoration followed.")


def eval_responder(prompt: str) -> str:
    """A scripted evaluation LLM: summaries for one prompt, scores for the other."""
    if "event summary robot" in prompt:
        return OTHER_DIM if "Revolutions of 1848" in prompt else FOUR_DIM
    return "3"


def make_context(mock_search, responses=eval_responder, cache=None):
    return EvaluationContext(
        llm=MockLLMProvider(responses=responses, model="mock-eval-model"),
        wiki=WikipediaHelper(mock_search, max_chars=4096),
        cache=cache,
    )


class StubMethodRunner(MethodRunner):
    """Returns canned answers instead of calling the real methods."""

    def __init__(self, answers=None, fail_with=None):
        super().__init__(GenerationConfig())
        self.answers = answers or {}
        self.fail_with = fail_with
        self.calls = []

    def run(self, method, event):
        method = canonical(method)
        self.calls.append((method, event.get("event_name")))
        if self.fail_with is not None:
            raise self.fail_with
        answer = self.answers.get(method, "Revolutions of 1848")
        extra = {}
        if method == "agentic":
            extra = {"final_winning_candidate": answer, "refinement_rounds": 2,
                     "judge_ranking": [{"rank": 1, "event_name": answer, "score": 8}]}
        return {"analogy_event": answer, "candidate": [["A", "B"]], "extra": extra}


@pytest.fixture
def popular_events():
    return [
        {"event_name": "Arab Spring", "event_intro": "A wave of uprisings.",
         "target_event": "Revolutions of 1989"},
        {"event_name": "Capitol Hill riot", "event_intro": "A storming of a capitol.",
         "target_event": "Storming of the Bastille"},
    ]


@pytest.fixture
def eval_run(tmp_path, mock_search, popular_events, monkeypatch):
    """An AutomaticEvaluation wired to stub methods and a fixed dataset."""
    def _build(methods=("agentic",), answers=None, fail_with=None, **overrides):
        monkeypatch.setattr("automatic_evaluation.runner.load_dataset",
                            lambda name: list(popular_events))
        overrides.setdefault("output_dir", tmp_path)
        config = EvaluationConfig(methods=list(methods), **overrides)
        return AutomaticEvaluation(
            config,
            evaluation_context=make_context(mock_search),
            method_runner=StubMethodRunner(answers, fail_with),
        ), config
    return _build


# ==========================================================================
# The metric itself -- must match the original evaluation.py
# ==========================================================================
def test_paper_constants():
    assert DIMENSIONS == ("topic", "background", "process", "result")
    assert DIMENSION_WEIGHTS == {"topic": 0.5, "background": 1.0,
                                 "process": 2.0, "result": 2.0}
    assert LITERAL_THRESHOLD == 0.35


def test_literal_similarity_matches_the_original_jacc():
    """Reimplement evaluation.py::jacc inline and compare."""
    from hal.text_similarity import content_tokens

    def original_jacc(text1, text2):
        set1, set2 = content_tokens(text1), content_tokens(text2)
        union = len(set1 | set2)
        return len(set1 & set2) / union if union else 0

    pairs = [("global pandemic outbreak", "global influenza outbreak"),
             ("the cold war", "a cold war"),
             ("revolution in france", "pandemic in asia"),
             ("", "")]
    for a, b in pairs:
        assert jacc(a, b) == pytest.approx(original_jacc(a, b))


def test_mds_matches_the_original_aggregation_formula():
    """Compare against evaluation.py's overall_score['all'] computation."""
    scores = {
        "topic": {"abstract_level": 4, "literal_level": 0.10},
        "background": {"abstract_level": 3, "literal_level": 0.20},
        "process": {"abstract_level": 2, "literal_level": 0.40},   # >= alpha -> 0
        "result": {"abstract_level": 3, "literal_level": 0.00},
    }
    # --- inline copy of the original loop body -------------------------
    overall_temp = {}
    for d in ("topic", "background", "process", "result"):
        abstract_level = scores[d]["abstract_level"]
        literal_level = scores[d]["literal_level"]
        if literal_level >= 0.35:
            literal_level = 0
        else:
            literal_level = 0.35 - literal_level
        overall_temp[d] = abstract_level * literal_level
    expected = (overall_temp["topic"] * 0.5 + overall_temp["background"] * 1
                + overall_temp["process"] * 2 + overall_temp["result"] * 2)
    # -------------------------------------------------------------------
    assert mds_from_scores(scores) == pytest.approx(expected)

    flat = component_scores(scores)
    assert flat["MDS"] == pytest.approx(expected)
    # *All is the UNWEIGHTED per-dimension product, as in overall_score[d]
    assert flat["TAll"] == pytest.approx(overall_temp["topic"])
    assert flat["PAll"] == 0.0            # literal similarity hit the threshold
    assert flat["TAbs"] == 4 and flat["TLit"] == pytest.approx(0.10)


def test_component_columns_are_complete():
    scores = {d: {"abstract_level": 2, "literal_level": 0.1} for d in DIMENSIONS}
    flat = component_scores(scores)
    assert set(COMPONENT_COLUMNS) == set(flat)


def test_alpha_threshold_zeroes_a_literal_match():
    scores = {d: {"abstract_level": 4, "literal_level": 0.9} for d in DIMENSIONS}
    assert mds_from_scores(scores) == 0.0


# ==========================================================================
# Four-dimension parsing + scoring one sample
# ==========================================================================
def test_four_dimension_extraction(mock_search):
    context = make_context(mock_search)
    event = extract_features({"event_name": "Arab Spring", "event_intro": "x"},
                             context)
    assert event["topic"].strip() == "A wave of uprisings."
    assert event["result"].strip() == "Some regimes fell."


def test_score_sample_reports_status(mock_search):
    context = make_context(mock_search)
    sample = {"event_name": "Arab Spring", "event_intro": "x",
              "analogy_event": "Revolutions of 1848"}
    row, status = score_sample_detailed(sample, context)
    assert status == "ok"
    assert set(row["score"]) == set(DIMENSIONS)
    assert row["mds"] > 0

    row, status = score_sample_detailed({**sample, "analogy_event": ""}, context)
    assert (row, status) == (None, "no_analogy")

    row, status = score_sample_detailed({**sample, "analogy_event": "Nonexistent"},
                                        context)
    assert (row, status) == (None, "unresolved_event")


# ==========================================================================
# Pass@1
# ==========================================================================
def test_pass_1_uses_search_matching_not_string_equality(mock_search):
    context = make_context(mock_search)
    assert pass_1_single("Cold War", "Cold War", context) is True
    assert pass_1_single("Cold War", "Spanish flu", context) is False


def test_pass_1_appears_only_for_the_popular_dataset(eval_run, popular_events):
    evaluation, _ = eval_run(methods=["agentic"])
    row = evaluation.evaluate_example("popular", 0, popular_events[0], "agentic")
    assert "pass_1" in row

    general_event = {"event_name": "Western Front", "event_intro": "a theatre",
                     "event_type": "War"}
    row = evaluation.evaluate_example("general", 0, general_event, "agentic")
    assert "pass_1" not in row       # no reference answer -> no Pass@1


def test_aggregate_omits_pass_1_for_general(eval_run, popular_events):
    evaluation, _ = eval_run(methods=["agentic"])
    rows = [evaluation.evaluate_example("general", i,
                                        {"event_name": e["event_name"],
                                         "event_intro": e["event_intro"]},
                                        "agentic")
            for i, e in enumerate(popular_events)]
    summary = aggregate(rows, "general")
    assert summary[0]["Pass@1"] is None
    assert summary[0]["n_evaluated"] == 2


# ==========================================================================
# Method registry -- reuses the existing implementations
# ==========================================================================
def test_method_names_and_aliases():
    assert canonical("self_reflection") == "reflection_generation"
    assert canonical("summarizing") == "summary_generation"
    assert canonical("two_stage_retrieval") == "twostage_retrieval"
    assert expand_methods(["all"]) == ALL_METHODS
    assert expand_methods(["baselines"]) == ALL_METHODS[:-1]
    assert expand_methods(["direct_generation", "agentic"]) == [
        "direct_generation", "agentic"]
    with pytest.raises(ValueError):
        canonical("not_a_method")


def test_all_seven_methods_are_registered():
    assert set(METHOD_LABELS) == {
        "direct_retrieval", "twostage_retrieval", "direct_generation",
        "twostage_generation", "summary_generation", "reflection_generation",
        "agentic",
    }


def test_registry_delegates_to_the_real_implementations(monkeypatch, mock_search,
                                                        offline_settings):
    """The evaluator must call the existing method, not reimplement it."""
    called = {}

    def fake_run(event, context):
        called["direct_generation"] = event["event_name"]
        return {"event_name": event["event_name"], "analogy_event": "Spanish flu",
                "candidate": []}

    import gemini_baselines
    monkeypatch.setattr(gemini_baselines, "get_method", lambda name: fake_run)
    runner = MethodRunner(GenerationConfig())
    monkeypatch.setattr(runner, "baseline_context", lambda: object())
    out = runner.run("direct_generation", {"event_name": "Arab Spring"})
    assert called["direct_generation"] == "Arab Spring"
    assert out["analogy_event"] == "Spanish flu"


# ==========================================================================
# Agentic: only the FINAL analogy is scored
# ==========================================================================
def test_agentic_final_answer_is_what_gets_scored(eval_run, popular_events):
    evaluation, _ = eval_run(methods=["agentic"],
                             answers={"agentic": "Revolutions of 1848"})
    row = evaluation.evaluate_example("popular", 0, popular_events[0], "agentic")
    assert row["predicted_analogy"] == "Revolutions of 1848"
    assert row["status"] == "ok"


def test_agentic_metadata_is_recorded_but_does_not_change_the_score(eval_run,
                                                                    popular_events):
    evaluation, _ = eval_run(methods=["agentic", "direct_generation"],
                             answers={"agentic": "Revolutions of 1848",
                                      "direct_generation": "Revolutions of 1848"})
    agentic = evaluation.evaluate_example("popular", 0, popular_events[0], "agentic")
    baseline = evaluation.evaluate_example("popular", 0, popular_events[0],
                                           "direct_generation")
    assert agentic["method_metadata"]["refinement_rounds"] == 2
    assert agentic["method_metadata"]["judge_ranking"][0]["rank"] == 1
    assert "method_metadata" not in baseline
    # identical answer -> identical score: no advantage for the agentic method
    for column in COMPONENT_COLUMNS:
        assert agentic[column] == pytest.approx(baseline[column])


# ==========================================================================
# Caching
# ==========================================================================
def test_repeated_events_are_summarised_only_once(tmp_path, mock_search):
    from hal.cache import JsonCache

    cache = JsonCache(tmp_path, "eval-cache")
    context = make_context(mock_search, cache=cache)
    sample = {"event_name": "Arab Spring", "event_intro": "x",
              "analogy_event": "Revolutions of 1848"}
    score_sample_detailed(dict(sample), context)
    calls_after_first = context.llm_calls
    score_sample_detailed(dict(sample), context)
    assert context.llm_calls == calls_after_first   # everything served from cache
    assert context.cache_hits > 0


def test_cache_is_keyed_by_evaluation_model(tmp_path, mock_search):
    from hal.cache import JsonCache

    cache = JsonCache(tmp_path, "eval-cache")
    first = make_context(mock_search, cache=cache)
    second = make_context(mock_search, cache=cache)
    second.llm.model = "a-different-model"

    abstract_similarity("a", "b", first)
    hits_before = second.cache_hits
    abstract_similarity("a", "b", second)
    assert second.cache_hits == hits_before      # a different model must not reuse


def test_cache_never_changes_the_result(tmp_path, mock_search):
    from hal.cache import JsonCache

    sample = {"event_name": "Arab Spring", "event_intro": "x",
              "analogy_event": "Revolutions of 1848"}
    uncached, _ = score_sample_detailed(dict(sample), make_context(mock_search))
    cached_context = make_context(mock_search, cache=JsonCache(tmp_path, "c"))
    score_sample_detailed(dict(sample), cached_context)          # populate
    cached, _ = score_sample_detailed(dict(sample), cached_context)
    assert cached["mds"] == pytest.approx(uncached["mds"])


# ==========================================================================
# Output files
# ==========================================================================
def test_detailed_jsonl_is_written_incrementally(eval_run):
    evaluation, config = eval_run(methods=["agentic"])
    evaluation.run_dataset("popular")
    path = _detailed_path(config, "popular")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["dataset"] == "popular"
        assert row["method"] == "agentic"
        assert row["evaluation_model"] == "mock-eval-model"
        assert "MDS" in row and "TAbs" in row
        assert "timestamp" in row and "alpha" in row
        assert row["dimension_weights"] == DIMENSION_WEIGHTS
        assert "api_key" not in json.dumps(row).lower()


def test_summary_csv_has_one_row_per_method(eval_run, tmp_path):
    evaluation, config = eval_run(methods=["agentic", "direct_generation"])
    rows = evaluation.run_dataset("popular")
    summary = aggregate(rows, "popular")
    path = write_summary_csv(summary, _summary_path(config, "popular"), True)
    with open(path, encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert {r["method"] for r in table} == {"agentic", "direct_generation"}
    for column in COMPONENT_COLUMNS + ["Pass@1"]:
        assert column in table[0]


def test_methods_that_never_ran_still_appear_in_the_table(eval_run):
    """A quota stop must not make a method silently vanish from the results."""
    evaluation, _ = eval_run(methods=["direct_generation"])
    rows = evaluation.run_dataset("popular")
    summary = aggregate(rows, "popular", methods=["direct_generation", "agentic"])
    by_method = {entry["method"]: entry for entry in summary}
    assert set(by_method) == {"direct_generation", "agentic"}
    assert by_method["agentic"]["n_evaluated"] == 0
    assert by_method["agentic"]["MDS"] is None


def test_table_warns_about_methods_that_did_not_run(capsys, eval_run):
    evaluation, _ = eval_run(methods=["direct_generation"])
    rows = evaluation.run_dataset("popular")
    summary = aggregate(rows, "popular", methods=["direct_generation", "agentic"])
    print_summary_table(summary, "popular", True)
    out = capsys.readouterr().out
    assert "NOT RUN" in out
    assert "this table is incomplete" in out
    assert "Our Agentic Pipeline" in out
    assert "--resume" in out


def test_table_warns_when_answers_could_not_be_scored(capsys, eval_run,
                                                      popular_events):
    evaluation, _ = eval_run(methods=["agentic"], answers={"agentic": "Nonexistent"})
    row = evaluation.evaluate_example("popular", 0, popular_events[0], "agentic")
    print_summary_table(aggregate([row], "popular", methods=["agentic"]),
                        "popular", True)
    out = capsys.readouterr().out
    assert "could not be scored" in out
    assert "unresolved_event" in out


def test_quota_notice_explains_the_daily_cap(capsys):
    from automatic_evaluation.runner import _print_quota_notice

    daily = ("429 RESOURCE_EXHAUSTED ... quotaId: "
             "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    _print_quota_notice(RuntimeError(daily), "popular", "summary_generation", 7)
    out = capsys.readouterr().out
    assert "PER-DAY" in out
    assert "--resume" in out
    assert "summary_generation" in out and "index=7" in out


def test_summary_table_prints(capsys, eval_run):
    evaluation, _ = eval_run(methods=["agentic"])
    rows = evaluation.run_dataset("popular")
    print_summary_table(aggregate(rows, "popular"), "popular", True, smoke=True)
    out = capsys.readouterr().out
    assert "AUTOMATIC EVALUATION -- POPULAR" in out
    assert "SMOKE RESULTS" in out
    assert "MDS" in out


def test_smoke_results_are_written_to_separate_files(eval_run):
    _, config = eval_run(methods=["agentic"], smoke=True)
    assert "_smoke" in _detailed_path(config, "popular").name
    assert "_smoke" in _summary_path(config, "popular").name


# ==========================================================================
# Resume
# ==========================================================================
def test_resume_skips_completed_pairs(eval_run):
    evaluation, config = eval_run(methods=["agentic"])
    evaluation.run_dataset("popular")
    first_calls = len(evaluation.methods.calls)
    assert first_calls == 2

    resumed, config2 = eval_run(methods=["agentic"], resume=True,
                                output_dir=config.output_dir)
    resumed.run_dataset("popular")
    assert resumed.methods.calls == []          # nothing regenerated
    rows = load_completed(_detailed_path(config, "popular"))
    assert len(rows) == 2


def test_resume_reuses_saved_generations(eval_run):
    evaluation, config = eval_run(methods=["agentic"])
    evaluation.run_dataset("popular")
    assert _generations_path(config, "popular", "agentic").exists()

    # wipe the scores but keep the saved answers -> generation must not rerun
    _detailed_path(config, "popular").unlink()
    resumed, _ = eval_run(methods=["agentic"], resume=True,
                          output_dir=config.output_dir)
    rows = resumed.run_dataset("popular")
    assert resumed.methods.calls == []
    assert len(rows) == 2


def test_evaluate_only_uses_saved_answers(eval_run):
    evaluation, config = eval_run(methods=["agentic"])
    evaluation.run_dataset("popular", evaluate=False, generate=True)
    assert len(evaluation.methods.calls) == 2

    scorer, _ = eval_run(methods=["agentic"], output_dir=config.output_dir)
    rows = scorer.run_dataset("popular", evaluate=True, generate=False)
    assert scorer.methods.calls == []           # no method was re-run
    assert len(rows) == 2 and all(r["status"] == "ok" for r in rows)


def test_generate_only_writes_answers_without_scoring(eval_run):
    evaluation, config = eval_run(methods=["agentic"])
    evaluation.run_dataset("popular", evaluate=False, generate=True)
    assert _generations_path(config, "popular", "agentic").exists()
    assert not _detailed_path(config, "popular").exists()


# ==========================================================================
# Provider failures
# ==========================================================================
def test_quota_error_is_recognised():
    assert is_quota_error(ProviderError("429 RESOURCE_EXHAUSTED: quota"))
    assert is_quota_error(RuntimeError("Rate limit exceeded"))
    assert not is_quota_error(ValueError("bad json"))


def test_quota_failure_preserves_earlier_results(eval_run, monkeypatch):
    evaluation, config = eval_run(methods=["agentic"])
    original_run = evaluation.methods.run
    calls = {"n": 0}

    def failing_run(method, event):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ProviderError("429 RESOURCE_EXHAUSTED: quota exceeded")
        return original_run(method, event)

    monkeypatch.setattr(evaluation.methods, "run", failing_run)
    rows = evaluation.run_dataset("popular")
    assert len(rows) == 1                       # the first example survived
    assert rows[0]["status"] == "ok"
    saved = load_completed(_detailed_path(config, "popular"))
    assert len(saved) == 1                      # and it is on disk


def test_non_quota_provider_error_is_recorded_not_scored(eval_run, popular_events):
    evaluation, _ = eval_run(methods=["agentic"],
                             fail_with=ProviderError("model not found"))
    row = evaluation.evaluate_example("popular", 0, popular_events[0], "agentic")
    assert row["status"] == "error"
    assert "model not found" in row["error"]
    # a failed call must never be given a fake score
    assert not any(column in row for column in COMPONENT_COLUMNS)


def test_unscorable_examples_are_excluded_from_averages(eval_run, popular_events):
    evaluation, _ = eval_run(methods=["agentic"], answers={"agentic": "Nonexistent"})
    row = evaluation.evaluate_example("popular", 0, popular_events[0], "agentic")
    assert row["status"] == "unresolved_event"
    summary = aggregate([row], "popular")
    assert summary[0]["n_evaluated"] == 0
    assert summary[0]["n_attempted"] == 1
    assert summary[0]["MDS"] is None
    # Pass@1 still counts it as a miss, as the original divides by len(dataset)
    assert summary[0]["Pass@1"] == 0.0
