"""This module contains the logic to build LLM chains."""

from collections.abc import Callable
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from src.fewshots.fewshooter_builder import create_dynamic_fewshooter
from src.utils import get_llm


def build_prompt(
    system_prompt: str,
    k: int = 5,
    group: str | None = None,
    yaml_path: Path | None = None,
) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate with dynamic few-shot selection."""
    few_shooter = create_dynamic_fewshooter(k=k, group=group, yaml_path=yaml_path)
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "system",
                "## A continuación, ejemplos de requerimientos y respuestas parecidas:",
            ),
            few_shooter,
            ("system", "## A continuación, el requerimiento del usuario:"),
            ("human", "{input}"),
        ]
    )


def build_structured_chain(
    *,
    system_prompt: str,
    output_schema: type[BaseModel],
    k: int = 5,
    temperature: float = 0,
    postprocess: Callable | None = None,
    group: str | None = None,
    yaml_path: Path | None = None,
) -> Runnable:
    """Create a structured-output chain for an arbitrary system prompt and schema."""
    llm = get_llm().bind(temperature=temperature)
    prompt = build_prompt(
        system_prompt=system_prompt,
        k=k,
        group=group,
        yaml_path=yaml_path,
    )
    pipeline: Runnable = prompt | llm.with_structured_output(output_schema)
    if postprocess is not None:
        pipeline = pipeline | RunnableLambda(postprocess)
    return pipeline.with_retry(stop_after_attempt=3)
