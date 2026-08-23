"""Core evaluation engine: analogy generation, deliberation, forecasting."""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from analysis.schemas import AnalysisPacket, ForecastRecord, RunManifest
from runner.forecastbench import BenchmarkQuestion
from runner.prompts_util import (
    deliberation_json_to_text,
    format_analogy_packet,
    format_deliberation_prompt,
    format_forecast_prompt,
    load_prompt,
)

ROOT = Path(__file__).resolve().parent.parent


def _setup_hal_repo(repo_path: Path) -> None:
    resolved = (ROOT / repo_path).resolve()
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))


def _parse_forecast_response(text: str, hal_repo: Optional[Path] = None) -> tuple[Optional[float], str]:
    import json as _json

    data = None
    try:
        data = _json.loads(text)
    except _json.JSONDecodeError:
        if hal_repo is not None:
            _setup_hal_repo(hal_repo)
            try:
                from hal.json_utils import parse_json_object
                data = parse_json_object(text or "")
            except Exception:
                data = None
    if not isinstance(data, dict):
        return None, ""
    try:
        p = float(data.get("p_yes"))
    except (TypeError, ValueError):
        return None, str(data.get("rationale") or "")
    p = max(0.0, min(1.0, p))
    return p, str(data.get("rationale") or "")


def _question_as_event(q: BenchmarkQuestion) -> Dict[str, str]:
    intro = (
        f"{q.background}\n\n"
        f"Forecast question: {q.question}\n\n"
        f"We seek historical events structurally analogous to this situation, "
        f"to inform a probabilistic forecast as of {q.forecast_timestamp}."
    )
    return {
        "event_name": q.question[:200],
        "event_intro": intro.strip(),
    }


def _read_completed_keys(path: Path) -> Set[str]:
    done: Set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add(f"{row['question_id']}:{row['condition']}")
    return done


def _load_analysis_packets(path: Path) -> Dict[str, str]:
    """Return question_id -> content for already-generated packets."""
    packets: Dict[str, str] = {}
    if not path.exists():
        return packets
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            packets[f"{row['question_id']}:{row['condition']}"] = row.get("content", "")
    return packets


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


class EvaluationEngine:
    def __init__(self, config: Dict[str, Any], run_id: str, dry_run: bool = False):
        self.config = config
        self.run_id = run_id
        self.dry_run = dry_run
        self.root = ROOT
        self.run_dir = self.root / config["output"]["runs_dir"] / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        prompts_cfg = config["prompts"]
        self.forecast_template = load_prompt(self.root, prompts_cfg["forecast"])
        self.delib_template = load_prompt(self.root, prompts_cfg["matched_deliberation"])
        self.wrapper_template = load_prompt(self.root, prompts_cfg["analysis_packet_wrapper"])
        self.analogy_format = load_prompt(self.root, prompts_cfg["analogy_packet_format"])

        self.model_name = config["model"]["llm"]
        self.temperature = float(config["model"].get("temperature", 0.1))
        self.max_output_tokens = int(config["model"].get("max_output_tokens", 4096))

        self._llm = None
        self._pipeline = None

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        if self.dry_run:
            from runner.mock_llm import MockLLM

            self._llm = MockLLM()
            return self._llm
        _setup_hal_repo(Path(self.config["analogy_pipeline"]["repo_path"]))
        from hal.config import get_settings
        from hal.providers.factory import get_llm

        get_settings()  # initialise singleton from environment / .env
        self._llm = get_llm(role="baseline")
        return self._llm

    def _get_pipeline(self, smoke: bool):
        if self._pipeline is not None:
            return self._pipeline
        if self.dry_run:
            return None
        _setup_hal_repo(Path(self.config["analogy_pipeline"]["repo_path"]))
        from agentic_pipeline.pipeline import AgenticAnalogyPipeline

        ap_cfg = self.config["analogy_pipeline"]
        overrides = {}
        if smoke:
            overrides = {"refinement_rounds": 1, "react_max_steps": 2, "verbose": True}
        else:
            overrides = {
                "refinement_rounds": ap_cfg.get("refinement_rounds", 2),
                "react_max_steps": ap_cfg.get("react_max_steps", 4),
                "verbose": True,
            }
        self._pipeline = AgenticAnalogyPipeline.build(**overrides)
        return self._pipeline

    def _llm_generate(self, prompt: str, *, json_output: bool = True) -> tuple[str, int, int, int]:
        if self.dry_run:
            if json_output and "p_yes" in self.forecast_template:
                if "base_rates" in prompt:
                    body = json.dumps({
                        "base_rates": "Historical base rate ~30%.",
                        "causal_drivers": "Institutional inertia.",
                        "uncertainties": "Timing unknown.",
                        "counterarguments": "Prior cycle differed.",
                        "time_horizon": "Short window.",
                    })
                else:
                    body = json.dumps({"p_yes": 0.42, "rationale": "Dry-run forecast."})
            else:
                body = '{"p_yes": 0.42, "rationale": "Dry-run."}'
            return body, 100, 50, 10

        llm = self._get_llm()
        t0 = time.time()
        text = llm.generate(
            prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            json_output=json_output,
        )
        ms = int((time.time() - t0) * 1000)
        return text or "", 0, 0, ms

    def generate_analogy_packet(self, q: BenchmarkQuestion, smoke: bool) -> tuple[str, dict]:
        if self.dry_run:
            content = (
                "**Winning analogy:** Velvet Revolution\n"
                "**Explanation:** Mass protests leading to regime change.\n"
            )
            meta = {"analogy_name": "Velvet Revolution", "dry_run": True}
            return content, meta

        pipeline = self._get_pipeline(smoke)
        event = _question_as_event(q)
        result = pipeline.run(event)
        content = format_analogy_packet(self.analogy_format, result)
        meta = result.to_dict() if hasattr(result, "to_dict") else {}
        return content, meta

    def generate_deliberation_packet(self, q: BenchmarkQuestion) -> tuple[str, dict]:
        qdict = q.to_dict()
        prompt = format_deliberation_prompt(
            self.delib_template, qdict, self.max_output_tokens
        )
        text, tin, tout, ms = self._llm_generate(prompt, json_output=True)
        import json as _json
        try:
            data = _json.loads(text)
        except _json.JSONDecodeError:
            data = {}
        content = deliberation_json_to_text(data)
        return content, {"raw": data, "latency_ms": ms}

    def forecast(
        self,
        q: BenchmarkQuestion,
        condition: str,
        analysis_packet: Optional[str] = None,
        packet_type: str = "none",
    ) -> ForecastRecord:
        qdict = q.to_dict()
        prompt = format_forecast_prompt(
            self.forecast_template,
            self.wrapper_template,
            qdict,
            analysis_packet=analysis_packet,
            packet_type=packet_type,
        )
        text, tin, tout, ms = self._llm_generate(prompt, json_output=True)
        p_yes, rationale = _parse_forecast_response(
            text,
            hal_repo=Path(self.config["analogy_pipeline"]["repo_path"]),
        )
        rec = ForecastRecord(
            question_id=q.question_id,
            condition=condition,
            p_yes=p_yes if p_yes is not None else 0.5,
            rationale=rationale,
            forecast_timestamp=q.forecast_timestamp,
            model=self.model_name,
            tokens_in=tin,
            tokens_out=tout,
            latency_ms=ms,
            outcome=q.outcome,
            round_id=q.round_id,
            cluster_id=q.cluster_id,
            run_id=self.run_id,
            status="ok" if p_yes is not None else "failed",
            error="" if p_yes is not None else "unparseable forecast",
        )
        return rec

    def run_question(
        self,
        q: BenchmarkQuestion,
        arms: List[str],
        *,
        smoke: bool,
        done: Set[str],
        cached_packets: Optional[Dict[str, str]] = None,
    ) -> None:
        packets: Dict[str, str] = {}
        cache = cached_packets or {}

        if "historical_analogy" in arms:
            cache_key = f"{q.question_id}:historical_analogy"
            if cache_key in cache:
                packets["historical_analogy"] = cache[cache_key]
            elif f"{q.question_id}:historical_analogy" not in done:
                print(f"  [{q.question_id}] generating analogy packet...")
                content, meta = self.generate_analogy_packet(q, smoke)
                packets["historical_analogy"] = content
                _append_jsonl(self.run_dir / "analysis_packets.jsonl", AnalysisPacket(
                    question_id=q.question_id,
                    condition="historical_analogy",
                    packet_type="analogy",
                    content=content,
                    model=self.model_name,
                    run_id=self.run_id,
                    metadata=meta,
                ).to_dict())

        if "matched_deliberation" in arms:
            cache_key = f"{q.question_id}:matched_deliberation"
            if cache_key in cache:
                packets["matched_deliberation"] = cache[cache_key]
            elif f"{q.question_id}:matched_deliberation" not in done:
                print(f"  [{q.question_id}] generating deliberation packet...")
                content, meta = self.generate_deliberation_packet(q)
                packets["matched_deliberation"] = content
                _append_jsonl(self.run_dir / "analysis_packets.jsonl", AnalysisPacket(
                    question_id=q.question_id,
                    condition="matched_deliberation",
                    packet_type="deliberation",
                    content=content,
                    model=self.model_name,
                    run_id=self.run_id,
                    metadata=meta,
                ).to_dict())

        order = list(arms)
        if self.config.get("compute", {}).get("randomize_arm_order", True):
            random.shuffle(order)

        for arm in order:
            key = f"{q.question_id}:{arm}"
            if key in done:
                continue
            print(f"  [{q.question_id}] forecasting ({arm})...")
            packet = packets.get(arm)
            ptype = arm if packet else "none"
            rec = self.forecast(q, arm, analysis_packet=packet, packet_type=ptype)
            _append_jsonl(self.run_dir / "forecasts.jsonl", rec.to_dict())
            print(f"    -> p_yes={rec.p_yes:.3f}  status={rec.status}")

    def write_manifest(self, n_questions: int, arms: List[str], smoke: bool) -> None:
        manifest = RunManifest(
            run_id=self.run_id,
            stage=self.config["study"]["stage"],
            config_path=str(self.config.get("_config_path", "")),
            started_at=datetime.now(timezone.utc).isoformat(),
            model=self.model_name,
            prompt_versions={
                "forecast": "forecast_v1",
                "matched_deliberation": "matched_deliberation_v1",
                "analogy_packet_format": "analogy_packet_format_v1",
            },
            n_questions=n_questions,
            arms=arms,
            status="running",
            notes=f"smoke={smoke} dry_run={self.dry_run}",
        )
        with (self.run_dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
