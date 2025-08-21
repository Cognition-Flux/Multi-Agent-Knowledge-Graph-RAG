"""This module contains the logic for the Cypher Query Agent."""

# %%
from __future__ import annotations

import json
import logging
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END
from langgraph.types import Command, Send

from KnowledgeGraphDB.Neo4j_KG_creation.cypher_runner import run_cypher
from src.agents.cypher_query_agent.llm_chains import (
    get_answer_generation_chain,
    get_cypher_query_chain,
)
from src.agents.cypher_query_agent.schemas import Neo4jQueryState
from src.agents.user_question_augmentation_agent.llm_chains import (
    get_question_generation_chain,
)
from src.utils import get_llm


# --- Setup ---
load_dotenv(override=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- LLM Chains ---
cypher_chain = get_cypher_query_chain(group="FEW_SHOTS_CYPHER_QUERY", k=5)
qgen_chain = get_question_generation_chain(group="FEW_SHOTS_QUESTIONS_GENERATION", k=5)
answer_chain = get_answer_generation_chain()


# --- Helper Functions ---
def safe_run_cypher(query: str) -> str:
    """Run Cypher query and return results as a JSON string.

    On error, return a JSON string with an 'error' key.
    """
    try:
        result = run_cypher(query)
        # Ensure results are JSON serializable
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logging.error("Error running Cypher query: %s", exc)
        return json.dumps({"error": f"ERROR: {exc}"}, ensure_ascii=False)


def _truncate_text(text: str, max_chars: int = 20000) -> str:
    """Truncate very long text to keep Bedrock payloads reasonable.

    Preserves the beginning and end of the string when truncation is applied,
    adding an omission note in the middle.
    """
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 200  # leave space for the omission note
    if tail < 0:
        tail = 0
    return (
        text[:head]
        + "\n\n[… contenido truncado por longitud; se omitieron partes …]\n\n"
        + text[-tail:]
    )


# --- Graph Nodes ---
async def generate_questions(
    state: Neo4jQueryState,
) -> Command[Literal["generate_cypher_queries_in_parallel"]]:
    """Node that generates alternative questions to augment the user's input."""
    logging.info("Generating alternative questions...")
    generated_questions = await qgen_chain.ainvoke({"input": state["question"]})
    return Command(
        goto="generate_cypher_queries_in_parallel",
        update={"generated_questions": generated_questions},
    )


async def generate_cypher_queries_in_parallel(
    state: Neo4jQueryState,
) -> Command[list[Send]]:
    """Node that fans out to generate a Cypher query for each augmented question."""
    logging.info("Generating Cypher queries in parallel for all questions...")
    lista_de_queries = [
        query.query_str for query in state["generated_questions"].queries_list
    ]
    sends = [
        Send(
            "generate_cypher_query",
            {"query": query},
        )
        for query in lista_de_queries
    ]
    return Command(goto=sends)


async def generate_cypher_query(
    state: Neo4jQueryState,
) -> Command[Literal["run_cypher_query_in_parallel"]]:
    """Node that generates a single Cypher query from an augmented question."""
    query_str = state["query"]
    logging.info("Generating Cypher query for: '%s'", query_str)
    cypher_res = await cypher_chain.ainvoke({"input": query_str})
    return Command(
        goto="run_cypher_query_in_parallel",
        update={"cypher_queries": cypher_res.cypher_query},
    )


async def run_cypher_query_in_parallel(
    state: Neo4jQueryState,
) -> Command[list[Send]]:
    """Node that fans out to execute each generated Cypher query."""
    logging.info("Executing all Cypher queries in parallel...")
    lista_de_cypher_queries = list(state["cypher_queries"])
    sends = [
        Send(
            "run_cypher_query",
            {"cypher_query": query},
        )
        for query in lista_de_cypher_queries
    ]
    return Command(goto=sends)


async def run_cypher_query(
    state: Neo4jQueryState,
) -> Command[Literal["generate_answer"]]:
    """Node that executes a single Cypher query and returns the result."""
    cypher_query = state["cypher_query"]
    logging.info("Running Cypher query: %s", cypher_query)
    results = safe_run_cypher(cypher_query)
    logging.info("Query results: %s", results)
    return Command(goto="generate_answer", update={"results": results})


async def generate_answer(
    state: Neo4jQueryState,
) -> Command[Literal[END]]:
    """Node that synthesizes a final answer from the aggregated query results."""
    logging.info("Generating the final answer...")
    question = state["question"]
    joined_results = "\n".join(state["results"]) if state.get("results") else ""
    truncated_results = _truncate_text(joined_results, max_chars=20000)
    input_for_llm = {
        "input": question,
        "number_of_results": len(state.get("results", [])),
        "results": truncated_results,
    }
    print("Input for generate_answer LLM: %s", input_for_llm)
    try:
        response = await answer_chain.ainvoke(input_for_llm)
        answer_text = getattr(response, "answer", None)
        if not answer_text:
            # Defensive: ensure we don't raise on missing field
            answer_text = str(response)
        return Command(goto=END, update={"messages": [AIMessage(content=answer_text)]})
    except Exception as exc:
        logger.warning(
            "Structured answer generation failed; using fallback. Error: %s", exc
        )
        # Fallback: plain chat without structured output
        fallback_system = (
            "Eres un asistente que sintetiza respuestas claras en formato markdown a partir de "
            "información proporcionada. Si la lista de resultados es muy larga, resume patrones y ejemplos, "
            "sin listar cada elemento."
        )
        fallback_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", fallback_system),
                (
                    "human",
                    "Question: {input}\n\nNumber of results: {number_of_results}\n\nResults (puede estar truncado):\n{results}\n\nGenera una respuesta completa en markdown.",
                ),
            ]
        )
        fallback_llm = get_llm()
        fallback_chain = fallback_prompt | fallback_llm
        msg = await fallback_chain.ainvoke(input_for_llm)
        content = getattr(msg, "content", str(msg))
        return Command(goto=END, update={"messages": [AIMessage(content=content)]})


# %%
