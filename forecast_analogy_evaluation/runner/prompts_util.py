"""Load and format frozen evaluation prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional


def load_prompt(root: Path, rel_path: str) -> str:
    return (root / rel_path).read_text(encoding="utf-8")


def _substitute(template: str, mapping: dict) -> str:
    """Replace {key} placeholders without interpreting other braces."""
    out = template
    for key, value in mapping.items():
        out = out.replace("{" + key + "}", str(value))
    return out


def format_deliberation_prompt(template: str, question: dict, max_tokens: int) -> str:
    return _substitute(template, {
        "question_text": question["question"],
        "background": question.get("background") or "(none)",
        "resolution_criteria": question.get("resolution_criteria") or "(none)",
        "forecast_timestamp": question.get("forecast_timestamp") or "",
        "max_output_tokens": max_tokens,
    })


def deliberation_json_to_text(data: dict) -> str:
    parts = []
    for key, label in [
        ("base_rates", "Base rates"),
        ("causal_drivers", "Causal drivers"),
        ("uncertainties", "Key uncertainties"),
        ("counterarguments", "Counterarguments"),
        ("time_horizon", "Time horizon"),
    ]:
        val = data.get(key)
        if val:
            parts.append(f"**{label}:**\n{val}")
    return "\n\n".join(parts)


def format_analogy_packet(template: str, result) -> str:
    evidence_urls = []
    if hasattr(result, "rounds"):
        for rnd in result.rounds:
            for rev in getattr(rnd, "revisions", []) or []:
                pass
    # Collect from agentic dict if available
    agentic = result.to_dict() if hasattr(result, "to_dict") else {}
    for rnd in agentic.get("rounds", []):
        for step in rnd.get("steps", []) if isinstance(rnd, dict) else []:
            pass

    sims = result.similarities if hasattr(result, "similarities") else []
    diffs = result.differences if hasattr(result, "differences") else []
    counters = result.counterexamples if hasattr(result, "counterexamples") else []
    limits = result.limitations if hasattr(result, "limitations") else []

    def bullet(items: List[str]) -> str:
        return "\n".join(f"- {x}" for x in items) if items else "(none listed)"

    return _substitute(template, {
        "analogy_name": result.analogy_event or "(none)",
        "explanation": result.explanation or "(none)",
        "similarities": bullet(sims),
        "differences": bullet(diffs),
        "counterexamples": bullet(counters),
        "limitations": bullet(limits),
        "evidence_urls": "(see run tool traces)",
    })


def format_forecast_prompt(
    template: str,
    wrapper: str,
    question: dict,
    analysis_packet: Optional[str] = None,
    packet_type: str = "none",
) -> str:
    if analysis_packet:
        packet_block = _substitute(wrapper, {
            "packet_type": packet_type,
            "packet_content": analysis_packet,
        })
    else:
        packet_block = ""

    return _substitute(template, {
        "question_text": question["question"],
        "background": question.get("background") or "(none)",
        "resolution_criteria": question.get("resolution_criteria") or "(none)",
        "forecast_timestamp": question.get("forecast_timestamp") or "",
        "analysis_packet": packet_block,
    })
