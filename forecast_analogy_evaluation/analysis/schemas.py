"""Forecast evaluation schemas.

Dataclasses for forecasts, analysis packets, and run manifests.
Kept independent from `hal/schemas.py` so forecast-specific types stay in this
subpackage, but both live in the same repository.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


CONDITIONS_MAIN = ("plain", "matched_deliberation", "historical_analogy")
CONDITIONS_DIAGNOSTIC = ("shuffled_analogy", "analogy_name_only")
CLIP_BOUNDS = (0.01, 0.99)


@dataclass
class ForecastRecord:
    question_id: str
    condition: str
    p_yes: float
    rationale: str = ""
    forecast_timestamp: str = ""
    prompt_version: str = "forecast_v1"
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    status: str = "ok"  # ok | failed | refused
    error: str = ""
    outcome: Optional[int] = None
    round_id: str = ""
    cluster_id: str = ""
    run_id: str = ""

    @property
    def p_yes_clipped(self) -> float:
        lo, hi = CLIP_BOUNDS
        return max(lo, min(hi, self.p_yes))

    def brier(self) -> Optional[float]:
        if self.outcome is None or self.status != "ok":
            return None
        p = self.p_yes_clipped
        y = float(self.outcome)
        return (p - y) ** 2

    def log_loss(self) -> Optional[float]:
        if self.outcome is None or self.status != "ok":
            return None
        import math
        p = self.p_yes_clipped
        y = float(self.outcome)
        return -(y * math.log(p) + (1 - y) * math.log(1 - p))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["p_yes_clipped"] = self.p_yes_clipped
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForecastRecord":
        return cls(
            question_id=str(data["question_id"]),
            condition=str(data["condition"]),
            p_yes=float(data["p_yes"]),
            rationale=str(data.get("rationale") or ""),
            forecast_timestamp=str(data.get("forecast_timestamp") or ""),
            prompt_version=str(data.get("prompt_version") or "forecast_v1"),
            model=str(data.get("model") or ""),
            tokens_in=int(data.get("tokens_in") or 0),
            tokens_out=int(data.get("tokens_out") or 0),
            latency_ms=int(data.get("latency_ms") or 0),
            status=str(data.get("status") or "ok"),
            error=str(data.get("error") or ""),
            outcome=data.get("outcome"),
            round_id=str(data.get("round_id") or ""),
            cluster_id=str(data.get("cluster_id") or ""),
            run_id=str(data.get("run_id") or ""),
        )


@dataclass
class AnalysisPacket:
    question_id: str
    condition: str
    packet_type: str  # deliberation | analogy | shuffled_analogy | analogy_name_only
    content: str
    prompt_version: str = ""
    model: str = ""
    tool_calls: int = 0
    tokens_out: int = 0
    run_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunManifest:
    run_id: str
    stage: str
    config_path: str
    started_at: str
    model: str
    prompt_versions: Dict[str, str] = field(default_factory=dict)
    n_questions: int = 0
    arms: List[str] = field(default_factory=list)
    status: str = "running"
    completed_at: str = ""
    notes: str = ""
    git_commit: str = ""
    environment: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
