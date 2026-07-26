"""Our agentic historical-analogy pipeline (the new method of this project).

Components, following the architecture diagram:

* :class:`~agentic_pipeline.generate_search_agent.GenerateSearchAgent` -- agent
* :class:`~agentic_pipeline.critic_agent.CriticAgent`                  -- agent
* :class:`~agentic_pipeline.anti_analogy_agent.AntiAnalogyAgent`       -- agent
* :class:`~agentic_pipeline.final_judge.FinalJudge`  -- ranking component after
  the loop; **not** an agent (no tools, no ReAct loop)
* :class:`~agentic_pipeline.final_summarizer.FinalSummarizer` -- final
  explanation stage
* :class:`~agentic_pipeline.pipeline.AgenticAnalogyPipeline` -- the loop itself
"""

from .anti_analogy_agent import AntiAnalogyAgent
from .critic_agent import CriticAgent
from .final_judge import FinalJudge, heuristic_ranking, rank_candidates
from .final_summarizer import FinalSummarizer
from .generate_search_agent import GenerateSearchAgent
from .pipeline import AgenticAnalogyPipeline, PipelineConfig, run_pipeline
from .react import ReActAgent, ReActOutcome, ReActStep

__all__ = [
    "GenerateSearchAgent",
    "CriticAgent",
    "AntiAnalogyAgent",
    "FinalJudge",
    "rank_candidates",
    "heuristic_ranking",
    "FinalSummarizer",
    "AgenticAnalogyPipeline",
    "PipelineConfig",
    "run_pipeline",
    "ReActAgent",
    "ReActOutcome",
    "ReActStep",
]
