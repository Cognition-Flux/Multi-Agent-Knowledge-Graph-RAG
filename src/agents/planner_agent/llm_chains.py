"""LLM chains for the planner agent.

uv run -m src.agents.planner_agent.llm_chains
"""

# %%
from pathlib import Path

import yaml
from langchain_core.runnables import Runnable

from src.agents.planner_agent.schemas import Plan
from src.chains.llm_chain_builder import build_structured_chain


_PROMPTS_PATH = Path(__file__).with_name("system_prompts.yaml")
with _PROMPTS_PATH.open(encoding="utf-8") as f:
    _data = yaml.safe_load(f) or {}
    _prompts = _data.get("LLM_CHAIN_SYSTEM_PROMPTS", {})
    SYSTEM_PROMPT_PLANNER_AGENT = _prompts.get(
        "SYSTEM_PROMPT_PLANNER_AGENT", ""
    ).strip()


def get_planner_chain(k: int = 5, group: str | None = "FEW_SHOTS_PLANNER") -> Runnable:
    """Convenience builder for the Cypher query agent chain."""
    return build_structured_chain(
        system_prompt=SYSTEM_PROMPT_PLANNER_AGENT,
        output_schema=Plan,
        k=k,
        temperature=0,
        group=group,
        yaml_path=Path(__file__).parent / "fewshots.yaml",
    )


if __name__ == "__main__":
    chain = get_planner_chain()
    response = chain.invoke(
        {"input": "resumen de los proyectos de la región de antofagasta"}
    )
    print(response.model_dump_json(indent=2))

# %%
