"""A small, transparent ReAct-style tool loop.

The presentation asks for ReAct behaviour inside the three search-based agents
(Generate/Search, Critic, Anti-Analogy): decide what evidence is needed, issue a
search, observe the result, update the answer, search again if necessary.

Implemented directly (no agent framework) so that every step is visible and
modifiable -- which matters more for research than framework features.

Protocol: at each step the model returns a single JSON object

    {"thought": "<one short sentence>",
     "action": "search" | "lookup" | "finish",
     "query": "<search query>",          # action = search
     "title": "<page title>",            # action = lookup
     "result": { ... }}                  # action = finish

What gets logged is observable only -- the query, the tool, the titles returned,
and the one-sentence rationale.  Long internal monologue is never requested,
stored, or printed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hal.json_utils import parse_json_object
from hal.providers.base import LLMProvider, SearchProvider
from hal.schemas import Evidence

MAX_THOUGHT_CHARS = 240
MAX_OBSERVATION_CHARS = 700


@dataclass
class ReActStep:
    """One observable step of the loop."""

    index: int
    action: str
    rationale: str = ""
    query: str = ""
    tool: str = ""
    results: List[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action,
            "rationale": self.rationale,
            "query": self.query,
            "tool": self.tool,
            "results": list(self.results),
            "note": self.note,
        }

    def describe(self) -> str:
        if self.action in ("search", "lookup"):
            found = ", ".join(self.results) if self.results else "nothing found"
            return f"{self.action}({self.query!r}) -> {found}"
        return f"{self.action}: {self.note or self.rationale}"


@dataclass
class ReActOutcome:
    """The result of a ReAct run."""

    result: Optional[Dict[str, Any]]
    steps: List[ReActStep] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.result is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result": self.result,
            "steps": [s.to_dict() for s in self.steps],
            "evidence": [e.to_dict() for e in self.evidence],
            "error": self.error,
        }


PROTOCOL = """You can use tools to gather evidence before answering.

Reply with exactly ONE JSON object per turn, and nothing else:

  To search an external knowledge base:
    {"thought": "<one short sentence naming the evidence you need>",
     "action": "search", "query": "<search query>"}

  To read one specific historical entry:
    {"thought": "<one short sentence>", "action": "lookup", "title": "<entry title>"}

  When you are ready to answer:
    {"thought": "<one short sentence>", "action": "finish", "result": <answer JSON>}

Rules:
- Keep "thought" to a single short sentence. Do not write long reasoning.
- Use at most __MAX_STEPS__ tool calls; you may finish earlier.
- Search only when the evidence would actually change your answer.
- The "result" object must follow the answer format described above.
"""


class ReActAgent:
    """Runs the loop for one task."""

    def __init__(self, llm: LLMProvider, search: Optional[SearchProvider] = None,
                 max_steps: int = 4, top_k: int = 4, verbose: bool = False,
                 name: str = "agent"):
        self.llm = llm
        self.search = search
        self.max_steps = max_steps
        self.top_k = top_k
        self.verbose = verbose
        self.name = name

    # -- tools ------------------------------------------------------------
    def _tool_search(self, query: str) -> (str, List[str], List[Evidence]):
        if self.search is None or not query:
            return "No search backend is available.", [], []
        results = self.search.search(query, top_k=self.top_k)
        if not results:
            return f"No results for {query!r}.", [], []
        evidence = [
            Evidence(query=query, title=r.title, snippet=r.snippet[:300], url=r.url,
                     tool="search")
            for r in results
        ]
        observation = "\n".join(r.brief(MAX_OBSERVATION_CHARS // max(len(results), 1))
                                for r in results)
        return observation, [r.title for r in results], evidence

    def _tool_lookup(self, title: str) -> (str, List[str], List[Evidence]):
        if self.search is None or not title:
            return "No search backend is available.", [], []
        page = self.search.get_page(title)
        if page is None:
            resolve = getattr(self.search, "resolve", None)
            page = resolve(title) if resolve else None
        if page is None:
            return (f"No entry named {title!r} exists. Treat it as unverified.",
                    [], [])
        evidence = [Evidence(query=title, title=page.title, snippet=page.snippet[:300],
                             url=page.url, tool="lookup")]
        return page.brief(MAX_OBSERVATION_CHARS), [page.title], evidence

    # -- loop -------------------------------------------------------------
    def run(self, task_prompt: str, *, system: Optional[str] = None) -> ReActOutcome:
        # PROTOCOL contains literal JSON braces, so substitute without str.format.
        protocol = PROTOCOL.replace("__MAX_STEPS__", str(self.max_steps))
        prompt = f"{task_prompt.rstrip()}\n\n{protocol}"
        transcript: List[str] = []
        steps: List[ReActStep] = []
        evidence: List[Evidence] = []
        tool_calls = 0

        for step_index in range(self.max_steps + 1):
            full_prompt = prompt
            if transcript:
                full_prompt += "\n\n== Evidence gathered so far ==\n" + "\n\n".join(transcript)
            if tool_calls >= self.max_steps:
                full_prompt += (
                    "\n\nYou have used all your tool calls. Reply now with the "
                    '{"action": "finish", "result": ...} JSON object.'
                )
            full_prompt += "\n\nYour JSON object:"

            raw = self.llm.generate(full_prompt, json_output=True, system=system) or ""
            data = parse_json_object(raw)
            if data is None:
                # One repair attempt, then give up on this step.
                repaired = self.llm.generate(
                    "Return only the JSON object described below, with no other text.\n\n"
                    f"Text to convert:\n{raw[:2000]}",
                    json_output=True,
                )
                data = parse_json_object(repaired or "")
            if data is None:
                steps.append(ReActStep(index=step_index, action="parse_error",
                                       note="model did not return usable JSON"))
                return ReActOutcome(result=None, steps=steps, evidence=evidence,
                                    error="unparseable model output")

            action = str(data.get("action", "finish")).lower().strip()
            rationale = str(data.get("thought", ""))[:MAX_THOUGHT_CHARS]

            if action == "finish" or "result" in data:
                result = data.get("result")
                if not isinstance(result, dict) or not result:
                    # Some models put the answer fields at the top level instead.
                    top_level = {k: v for k, v in data.items()
                                 if k not in ("thought", "action", "result")}
                    result = top_level or None
                steps.append(ReActStep(index=step_index, action="finish",
                                       rationale=rationale,
                                       note="answered" if result else "empty answer"))
                if self.verbose:
                    print(f"      [{self.name}] finish after {tool_calls} tool call(s)")
                return ReActOutcome(result=result, steps=steps, evidence=evidence,
                                    error="" if result
                                    else "finish without a result object")

            if tool_calls >= self.max_steps:
                steps.append(ReActStep(index=step_index, action="budget_exhausted",
                                       rationale=rationale))
                return ReActOutcome(result=None, steps=steps, evidence=evidence,
                                    error="tool budget exhausted without an answer")

            if action == "lookup":
                query = str(data.get("title") or data.get("query") or "")
                observation, titles, found = self._tool_lookup(query)
            else:
                action = "search"
                query = str(data.get("query") or data.get("title") or "")
                observation, titles, found = self._tool_search(query)

            tool_calls += 1
            evidence.extend(found)
            steps.append(ReActStep(index=step_index, action=action, rationale=rationale,
                                   query=query, tool=getattr(self.search, "name", ""),
                                   results=titles))
            if self.verbose:
                print(f"      [{self.name}] {action}({query!r}) -> "
                      f"{', '.join(titles) if titles else 'no results'}")
            transcript.append(f"{action}({query!r}):\n{observation}")

        return ReActOutcome(result=None, steps=steps, evidence=evidence,
                            error="loop ended without an answer")
