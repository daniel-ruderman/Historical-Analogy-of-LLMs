"""Structured objects exchanged between the pipeline components.

Plain dataclasses (no extra dependency) with ``to_dict``/``from_dict`` so every
run can be serialised to jsonl for later analysis.

The four dimensions of a historical event come from the paper: **topic,
background, process, result**.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

DIMENSIONS = ("topic", "background", "process", "result")


@dataclass
class EventDimensions:
    """The paper's four-dimensional description of an event."""

    topic: str = ""
    background: str = ""
    process: str = ""
    result: str = ""

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EventDimensions":
        data = data or {}
        return cls(
            topic=str(data.get("topic") or data.get("summary") or ""),
            background=str(data.get("background") or ""),
            process=str(data.get("process") or ""),
            result=str(data.get("result") or ""),
        )

    def is_empty(self) -> bool:
        return not any(getattr(self, d) for d in DIMENSIONS)

    def as_text(self) -> str:
        return (
            f"Topic: {self.topic}\nBackground: {self.background}\n"
            f"Process: {self.process}\nResult: {self.result}"
        )


@dataclass
class HistoricalEvent:
    """An event -- either the input event or a proposed historical analogy."""

    name: str
    description: str = ""
    dimensions: EventDimensions = field(default_factory=EventDimensions)
    source: str = ""          # "dataset" | "wikipedia" | "llm" | ...
    url: str = ""
    verified: Optional[bool] = None   # Wikipedia verification, when performed

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["dimensions"] = self.dimensions.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoricalEvent":
        return cls(
            name=str(data.get("name") or data.get("event_name") or ""),
            description=str(data.get("description") or data.get("event_intro") or ""),
            dimensions=EventDimensions.from_dict(data.get("dimensions")),
            source=str(data.get("source") or ""),
            url=str(data.get("url") or ""),
            verified=data.get("verified"),
        )

    @classmethod
    def from_dataset_row(cls, row: Dict[str, Any]) -> "HistoricalEvent":
        """Build from either dataset schema (analogy sets or the event pool)."""
        name = row.get("event_name") or row.get("history_event_text") or ""
        intro = row.get("event_intro") or row.get("history_intro_text") or ""
        return cls(name=name, description=intro, source="dataset",
                   url=row.get("url", ""))

    def text(self, max_chars: int = 1200) -> str:
        body = self.description
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "..."
        return f"{self.name}: {body}" if body else self.name


@dataclass
class Evidence:
    """A piece of retrieved supporting material (search result)."""

    query: str
    title: str
    snippet: str = ""
    url: str = ""
    tool: str = "search"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateAnalogy:
    """A proposed historical analogy for the input event."""

    event: HistoricalEvent
    rationale: str = ""                       # concise, user-facing rationale
    mapping: Dict[str, str] = field(default_factory=dict)  # dimension -> correspondence
    confidence: float = 0.5
    status: str = "new"                       # new | kept | revised | replaced
    origin_round: int = 0
    evidence: List[Evidence] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.event.name

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event.to_dict(),
            "rationale": self.rationale,
            "mapping": dict(self.mapping),
            "confidence": self.confidence,
            "status": self.status,
            "origin_round": self.origin_round,
            "evidence": [e.to_dict() for e in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateAnalogy":
        return cls(
            event=HistoricalEvent.from_dict(data.get("event", {})),
            rationale=str(data.get("rationale", "")),
            mapping=dict(data.get("mapping") or {}),
            confidence=float(data.get("confidence", 0.5) or 0.0),
            status=str(data.get("status", "new")),
            origin_round=int(data.get("origin_round", 0) or 0),
            evidence=[Evidence(**e) for e in data.get("evidence", []) if isinstance(e, dict)],
        )


@dataclass
class Critique:
    """The Critic agent's structured assessment of one candidate."""

    candidate: str
    dimension_scores: Dict[str, float] = field(default_factory=dict)  # topic/.../result, 1-4
    structural_correspondence: str = ""
    important_differences: List[str] = field(default_factory=list)
    weak_assumptions: List[str] = field(default_factory=list)
    factual_problems: List[str] = field(default_factory=list)
    surface_level: bool = False       # is it mostly a surface analogy?
    overall_score: float = 0.0        # 1-4, comparable to the paper's scale
    recommendation: str = "keep"      # keep | revise | replace
    summary: str = ""
    evidence: List[Evidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evidence"] = [e.to_dict() for e in self.evidence]
        return data

    def as_feedback(self) -> str:
        """Compact text handed back to the Generate/Search agent."""
        parts = [f"Candidate: {self.candidate}",
                 f"Overall (1-4): {self.overall_score:.1f} -> {self.recommendation}"]
        if self.dimension_scores:
            parts.append("Dimension scores: " + ", ".join(
                f"{k}={v}" for k, v in self.dimension_scores.items()))
        if self.summary:
            parts.append(f"Assessment: {self.summary}")
        if self.structural_correspondence:
            parts.append(f"Structural correspondence: {self.structural_correspondence}")
        if self.important_differences:
            parts.append("Important differences: " + "; ".join(self.important_differences))
        if self.weak_assumptions:
            parts.append("Weak assumptions: " + "; ".join(self.weak_assumptions))
        if self.factual_problems:
            parts.append("Factual problems: " + "; ".join(self.factual_problems))
        if self.surface_level:
            parts.append("Warning: judged to be mostly a surface-level analogy.")
        return "\n".join(parts)


@dataclass
class CounterExample:
    """One case found by the Anti-Analogy agent."""

    event_name: str
    description: str = ""
    divergence: str = ""       # how the outcome/mechanism diverges
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AntiAnalogyReport:
    """The Anti-Analogy agent's findings for one candidate."""

    candidate: str
    counterexamples: List[CounterExample] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    robustness: float = 0.5     # 0-1; low = the pattern often breaks down
    verdict: str = "holds"      # holds | weakened | undermined
    summary: str = ""
    evidence: List[Evidence] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["counterexamples"] = [c.to_dict() for c in self.counterexamples]
        data["evidence"] = [e.to_dict() for e in self.evidence]
        return data

    def as_feedback(self) -> str:
        parts = [f"Candidate: {self.candidate}",
                 f"Verdict: {self.verdict} (robustness {self.robustness:.2f})"]
        if self.summary:
            parts.append(self.summary)
        for counter in self.counterexamples:
            line = f"- Counterexample: {counter.event_name}"
            if counter.divergence:
                line += f" -- {counter.divergence}"
            parts.append(line)
        if self.failure_modes:
            parts.append("Failure modes: " + "; ".join(self.failure_modes))
        return "\n".join(parts)


@dataclass
class CandidateRevision:
    """What the Generate/Search agent did with a candidate in a round."""

    round_index: int
    action: str                # keep | revise | replace | drop | add
    candidate: str
    previous_candidate: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RefinementRound:
    """Everything observable that happened in one refinement round."""

    index: int
    critiques: List[Critique] = field(default_factory=list)
    anti_analogies: List[AntiAnalogyReport] = field(default_factory=list)
    revisions: List[CandidateRevision] = field(default_factory=list)
    candidates_after: List[CandidateAnalogy] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "critiques": [c.to_dict() for c in self.critiques],
            "anti_analogies": [a.to_dict() for a in self.anti_analogies],
            "revisions": [r.to_dict() for r in self.revisions],
            "candidates_after": [c.to_dict() for c in self.candidates_after],
        }


@dataclass
class RankedCandidate:
    """One row of the Final Judge's ranking."""

    rank: int
    event_name: str
    reason: str = ""
    weaknesses: List[str] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeRanking:
    """The Final Judge's output (a ranking component, not an agent)."""

    ranking: List[RankedCandidate] = field(default_factory=list)
    notes: str = ""

    @property
    def winner(self) -> Optional[RankedCandidate]:
        return self.ranking[0] if self.ranking else None

    def to_dict(self) -> Dict[str, Any]:
        return {"ranking": [r.to_dict() for r in self.ranking], "notes": self.notes}


@dataclass
class FinalAnalogyResult:
    """The complete result of one run of our agentic pipeline."""

    input_event: HistoricalEvent
    analogy_event: str = ""
    winning_candidate: Optional[CandidateAnalogy] = None
    explanation: str = ""
    similarities: List[str] = field(default_factory=list)
    differences: List[str] = field(default_factory=list)
    counterexamples: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    why_ranked_first: str = ""
    ranking: JudgeRanking = field(default_factory=JudgeRanking)
    initial_candidates: List[CandidateAnalogy] = field(default_factory=list)
    rounds: List[RefinementRound] = field(default_factory=list)
    final_candidates: List[CandidateAnalogy] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_event": self.input_event.to_dict(),
            "analogy_event": self.analogy_event,
            "winning_candidate": (
                self.winning_candidate.to_dict() if self.winning_candidate else None
            ),
            "explanation": self.explanation,
            "similarities": list(self.similarities),
            "differences": list(self.differences),
            "counterexamples": list(self.counterexamples),
            "limitations": list(self.limitations),
            "why_ranked_first": self.why_ranked_first,
            "ranking": self.ranking.to_dict(),
            "initial_candidates": [c.to_dict() for c in self.initial_candidates],
            "rounds": [r.to_dict() for r in self.rounds],
            "final_candidates": [c.to_dict() for c in self.final_candidates],
            "errors": list(self.errors),
        }

    def to_output_row(self) -> Dict[str, Any]:
        """Row in the original repository's output format.

        ``event_name``/``event_intro``/``analogy_event`` (+ ``candidate``) keeps
        the result directly usable by ``evaluation.py`` and by our provider-
        neutral MDS implementation.
        """
        return {
            "event_name": self.input_event.name,
            "event_intro": self.input_event.description,
            "analogy_event": self.analogy_event,
            "candidate": [[c.name for c in self.initial_candidates]]
            + [[c.name for c in r.candidates_after] for r in self.rounds],
            "agentic": self.to_dict(),
        }
