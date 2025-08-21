"""Create a chain for Cypher query generation.

- uv run -m src.agents.cypher_query_agent.llm_chains
"""

# %%
from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from KnowledgeGraphDB.Neo4j_KG_creation.cypher_runner import run_cypher
from src.agents.cypher_query_agent.schemas import (
    Answer,
    CypherQuery,
)
from src.chains.llm_chain_builder import build_structured_chain
from src.utils import get_llm


_PROMPTS_PATH = Path(__file__).with_name("system_prompts.yaml")
with _PROMPTS_PATH.open(encoding="utf-8") as f:
    _data = yaml.safe_load(f) or {}
    _prompts = _data.get("LLM_CHAIN_SYSTEM_PROMPTS", {})
    SYSTEM_PROMPT_CYPHER_QUERY_AGENT = _prompts.get(
        "SYSTEM_PROMPT_CYPHER_QUERY_AGENT", ""
    ).strip()
    SYSTEM_PROMPT_ANSWER_GENERATION_AGENT = _prompts.get(
        "SYSTEM_PROMPT_ANSWER_GENERATION_AGENT", ""
    ).strip()


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
        yaml_path=Path(__file__).parent / "fewshots.yaml",
    )


def get_answer_generation_chain() -> Runnable:
    """Convenience builder for an answer-generation agent chain.

    Modified to be compatible with Bedrock's structured output requirements.
    """
    llm = get_llm()

    # Enhanced system prompt that clearly instructs the model to generate an answer
    enhanced_system_prompt = (
        f"{SYSTEM_PROMPT_ANSWER_GENERATION_AGENT}\n\n"
        "IMPORTANTE: Debes generar una respuesta completa y detallada que responda a la pregunta del usuario "
        "basándote exclusivamente en los resultados proporcionados."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", enhanced_system_prompt),
            (
                "human",
                "Question: {input}\n\nNumber of results: {number_of_results}\n\nResults:\n{results}\n\nGenera una respuesta completa en formato markdown.",
            ),
        ]
    )
    pipeline: Runnable = prompt | llm.with_structured_output(Answer)
    return pipeline.with_retry(stop_after_attempt=3)


if __name__ == "__main__":
    cypher_chain = get_cypher_query_chain(group="FEW_SHOTS_CYPHER_QUERY", k=5)

    demo_input = {"input": "Comuna con más proyectos"}

    cypher_res = cypher_chain.invoke(demo_input)
    print("CypherQuery:")
    try:
        print(cypher_res.model_dump_json(indent=2))
    except Exception:
        print(cypher_res)

    result = run_cypher(cypher_res.cypher_query)
    print("Result:")
    print(result)
    # chain = get_answer_generation_chain()
    # res = chain.invoke(
    #     {
    #         "input": "Comuna con más proyectos",
    #         "results": "Antofagasta",
    #     }
    # )
    # print("Answer:")
    # try:
    #     print(res.model_dump_json(indent=2))
    # except Exception:
    #     print(res)
