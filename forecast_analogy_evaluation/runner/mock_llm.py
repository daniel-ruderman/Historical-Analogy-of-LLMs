"""Mock LLM for dry-run evaluation (no GPU/network)."""

from __future__ import annotations

import json


class MockLLM:
    name = "mock"
    model = "mock-llm"

    def generate(self, prompt, **kwargs) -> str:
        if "base_rates" in prompt:
            return json.dumps({
                "base_rates": "Roughly one third of similar cases resolve yes.",
                "causal_drivers": "Institutional momentum and public pressure.",
                "uncertainties": "Leadership decisions are hard to predict.",
                "counterarguments": "Prior cycle had different media environment.",
                "time_horizon": "Resolution window is short.",
            })
        return json.dumps({"p_yes": 0.42, "rationale": "Mock calibrated forecast."})
