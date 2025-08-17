"""This module contains the LLM chains for the user question augmentation agent."""

# %%
from __future__ import annotations

from pathlib import Path

import yaml
from langchain_core.runnables import Runnable

from src.agents.user_question_augmentation_agent.schemas import GeneratedQueries
from src.chains.llm_chain_builder import build_structured_chain


_PROMPTS_PATH = Path(__file__).with_name("system_prompts.yaml")
with _PROMPTS_PATH.open(encoding="utf-8") as f:
    _data = yaml.safe_load(f) or {}
    _prompts = _data.get("LLM_CHAIN_SYSTEM_PROMPTS", {})
    SYSTEM_PROMPT_QUESTION_GENERATION_AGENT = _prompts.get(
        "SYSTEM_PROMPT_QUESTION_GENERATION_AGENT", ""
    ).strip()


def _validate_queries(output: GeneratedQueries) -> GeneratedQueries:
    """Extra safeguard to ensure the output is a valid GeneratedQueries object."""
    if not isinstance(output, GeneratedQueries):
        raise TypeError("Output must be a GeneratedQueries object")
    if not output.queries_list:
        raise ValueError("Generated queries list cannot be empty")
    return output


def get_question_generation_chain(
    k: int = 5, group: str | None = "FEW_SHOTS_QUESTIONS_GENERATION"
) -> Runnable:
    """Convenience builder for a question-generation agent chain."""
    return build_structured_chain(
        system_prompt=SYSTEM_PROMPT_QUESTION_GENERATION_AGENT,
        output_schema=GeneratedQueries,
        k=k,
        temperature=0,
        postprocess=_validate_queries,
        group=group,
        yaml_path=Path(__file__).parent / "fewshots.yaml",
    )


if __name__ == "__main__":
    qgen_chain = get_question_generation_chain(
        group="FEW_SHOTS_QUESTIONS_GENERATION", k=5
    )
    demo_input = {"input": "proyectos en las comunas Antofagasta o Mejillones"}
    qgen_res = qgen_chain.invoke(demo_input)
    print("\nGeneratedQueries:")
    try:
        print(qgen_res.model_dump_json(indent=2))
    except Exception:
        print(qgen_res)

# %%
