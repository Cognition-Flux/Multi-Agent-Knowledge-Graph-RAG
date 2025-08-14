"""Create a chain for Cypher query generation.

- uv run -m src.agents.cypher_query_agent.llm_chains
"""

# %%
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

from src.agents.cypher_query_agent.fewshooter_builder import create_dynamic_fewshooter
from src.agents.cypher_query_agent.schemas import (
    CypherQuery,
    GeneratedQueries,
)
from src.utils import get_llm


_PROMPTS_PATH = Path(__file__).with_name("system_prompts.yaml")
with _PROMPTS_PATH.open(encoding="utf-8") as f:
    _data = yaml.safe_load(f) or {}
    _prompts = _data.get("LLM_CHAIN_SYSTEM_PROMPTS", {})
    SYSTEM_PROMPT_CYPHER_QUERY_AGENT = _prompts.get(
        "SYSTEM_PROMPT_CYPHER_QUERY_AGENT", ""
    ).strip()
    SYSTEM_PROMPT_QUESTION_GENERATION_AGENT = _prompts.get(
        "SYSTEM_PROMPT_QUESTION_GENERATION_AGENT", ""
    ).strip()


@lru_cache(maxsize=16)
def build_prompt(
    system_prompt: str, k: int = 5, group: str | None = None
) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate with dynamic few-shot selection.

    Caches by (system_prompt, k, group) to avoid rebuilding vectorstores repeatedly.
    """
    few_shooter = create_dynamic_fewshooter(k=k, group=group)
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
) -> Runnable:
    """Create a structured-output chain for an arbitrary system prompt and schema.

    If provided, `postprocess` will run after schema parsing and can raise to
    trigger retries (e.g., for additional business-rule validation).
    """
    llm = get_llm().bind(temperature=temperature)
    prompt = build_prompt(system_prompt=system_prompt, k=k, group=group)
    pipeline: Runnable = prompt | llm.with_structured_output(output_schema)
    if postprocess is not None:
        pipeline = pipeline | RunnableLambda(postprocess)
    return pipeline.with_retry(stop_after_attempt=3)


def _ensure_return_clause(output: CypherQuery) -> CypherQuery:
    """Additional safety check for read-style Cypher queries.

    Enforces presence of a RETURN clause to reduce chances of producing
    non-readable queries (e.g., accidental write-only queries). Raise to retry.
    """
    text = output.cypher_query.strip()
    if "return" not in text.lower():
        raise ValueError("cypher_query must contain a RETURN clause")
    return output


def get_cypher_query_chain(
    k: int = 5, group: str | None = "FEW_SHOTS_CYPHER_QUERY"
) -> Runnable:
    """Convenience builder for the Cypher query agent chain."""
    return build_structured_chain(
        system_prompt=SYSTEM_PROMPT_CYPHER_QUERY_AGENT,
        output_schema=CypherQuery,
        k=k,
        temperature=0,
        postprocess=_ensure_return_clause,
        group=group,
    )


def get_question_generation_chain(
    k: int = 5, group: str | None = "FEW_SHOTS_QUESTIONS_GENERATION"
) -> Runnable:
    """Convenience builder for a question-generation agent chain."""
    return build_structured_chain(
        system_prompt=SYSTEM_PROMPT_QUESTION_GENERATION_AGENT,
        output_schema=GeneratedQueries,
        k=k,
        temperature=0,
        group=group,
    )


SYSTEM_PROMPT_ANSWER_GENERATION_AGENT = """
 La pregunta es: {input} y la información para responderla es: {results}
 Genera una respuesta precisa, considerando exclusiva y únicamente la información proporcionada.
"""


class Answer(BaseModel):
    """Answer schema."""

    answer: str = Field(description="The answer to the question.")


def get_answer_generation_chain() -> Runnable:
    """Convenience builder for an answer-generation agent chain."""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_ANSWER_GENERATION_AGENT),
        ]
    )
    pipeline: Runnable = prompt | llm.with_structured_output(Answer)
    return pipeline.with_retry(stop_after_attempt=3)


if __name__ == "__main__":
    cypher_chain = get_cypher_query_chain(group="FEW_SHOTS_CYPHER_QUERY")
    qgen_chain = get_question_generation_chain(
        group="FEW_SHOTS_QUESTIONS_GENERATION", k=5
    )

    demo_input = {"input": "proyectos en las comunas Antofagasta o Mejillones"}

    cypher_res = cypher_chain.invoke(demo_input)
    print("CypherQuery:")
    try:
        print(cypher_res.model_dump_json(indent=2))
    except Exception:
        print(cypher_res)

    qgen_res = qgen_chain.invoke(demo_input)
    print("\nGeneratedQueries:")
    try:
        print(qgen_res.model_dump_json(indent=2))
    except Exception:
        print(qgen_res)
    chain = get_answer_generation_chain()
    res = chain.invoke(
        {
            "input": "donde se ubican los proyectos?",
            "results": "proyectos en las comunas Antofagasta o Mejillones",
        }
    )
    print("Answer:")
    try:
        print(res.model_dump_json(indent=2))
    except Exception:
        print(res)
