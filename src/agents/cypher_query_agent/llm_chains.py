"""Create a chain for Cypher query generation.

- uv run -m src.agents.cypher_query_agent.llm_chains
"""

# %%
from collections.abc import Callable
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel

from src.agents.cypher_query_agent.fewshooter_builder import create_dynamic_fewshooter
from src.agents.cypher_query_agent.schemas import (
    CypherQuery,
    GeneratedQueries,
)
from src.utils import get_llm


SYSTEM_PROMPT_CYPHER_QUERY_AGENT = "You are an expert Cypher query writer."
SYSTEM_PROMPT_QUESTION_GENERATION_AGENT = "You are an expert question generation agent."


@lru_cache(maxsize=16)
def build_prompt(system_prompt: str, k: int = 3) -> ChatPromptTemplate:
    """Build a ChatPromptTemplate with dynamic few-shot selection.

    Caches by (system_prompt, k) to avoid rebuilding vectorstores repeatedly.
    """
    few_shooter = create_dynamic_fewshooter(k=k)
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
    k: int = 3,
    temperature: float = 0,
    postprocess: Callable | None = None,
) -> Runnable:
    """Create a structured-output chain for an arbitrary system prompt and schema.

    If provided, `postprocess` will run after schema parsing and can raise to
    trigger retries (e.g., for additional business-rule validation).
    """
    llm = get_llm().bind(temperature=temperature)
    prompt = build_prompt(system_prompt=system_prompt, k=k)
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


def get_cypher_query_chain(k: int = 3) -> Runnable:
    """Convenience builder for the Cypher query agent chain."""
    return build_structured_chain(
        system_prompt=SYSTEM_PROMPT_CYPHER_QUERY_AGENT,
        output_schema=CypherQuery,
        k=k,
        temperature=0,
        postprocess=_ensure_return_clause,
    )


def get_question_generation_chain(k: int = 3) -> Runnable:
    """Convenience builder for a question-generation agent chain."""
    return build_structured_chain(
        system_prompt=SYSTEM_PROMPT_QUESTION_GENERATION_AGENT,
        output_schema=GeneratedQueries,
        k=k,
        temperature=0,
    )


if __name__ == "__main__":
    cypher_chain = get_cypher_query_chain()
    qgen_chain = get_question_generation_chain()

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
