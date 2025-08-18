"""Hybrid GraphRAG agent logic.

This module contains the asynchronous logic for the hybrid GraphRAG agent,
which generates multiple question variations and searches them in parallel
using the neo4j_graphrag library.
"""

# %%
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from langgraph.graph import END
from langgraph.types import Command, Send

from src.agents.cypher_query_agent.schemas import Neo4jQueryState
from src.agents.hybrid_graphRAG_agent.knowledge_graph_search import graph_rag
from src.agents.user_question_augmentation_agent.llm_chains import (
    get_question_generation_chain,
)


# --- Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- LLM Chains ---
qgen_chain = get_question_generation_chain(group="FEW_SHOTS_QUESTIONS_GENERATION", k=5)


# --- Helper Functions ---
async def async_graph_rag_search(
    query: str,
    retriever_config: dict[str, Any] | None = None,
    return_context: bool = False,
) -> Any:
    """Wrapper to run the synchronous graph_rag.search() asynchronously.

    Uses asyncio.to_thread to avoid blocking the event loop.

    Args:
        query: The query string to search for.
        retriever_config: Configuration for the retriever (e.g., top_k).
        return_context: Whether to return the context along with the answer.

    Returns:
        The response from graph_rag.search().

    Raises:
        Exception: If the GraphRAG search fails.
    """
    try:
        # Run the synchronous search in a thread pool to avoid blocking
        return await asyncio.to_thread(
            graph_rag.search,
            query,
            retriever_config=retriever_config or {"top_k": 5},
            return_context=return_context,
        )
    except Exception as exc:
        logging.error("GraphRAG search failed for query '%s': %s", query, exc)
        raise


# --- Graph Nodes ---
async def generate_questions(
    state: Neo4jQueryState,
) -> Command[Literal["send_queries_in_parallel"]]:
    """Node that generates alternative questions to augment the user's input.

    This node uses the question generation chain to create multiple variations
    of the original question for more comprehensive search coverage.

    Args:
        state: The current graph state containing the original question.

    Returns:
        Command directing to send_queries_in_parallel with generated questions.
    """
    logging.info("Generating alternative questions for: '%s'", state["question"])
    try:
        generated_questions = await qgen_chain.ainvoke({"input": state["question"]})
        logging.info(
            "Generated %d alternative questions", len(generated_questions.queries_list)
        )
        return Command(
            goto="send_queries_in_parallel",
            update={"generated_questions": generated_questions},
        )
    except Exception as exc:
        logging.error("Failed to generate questions: %s", exc)
        # Fallback: use only the original question
        from src.agents.cypher_query_agent.schemas import GeneratedQueries, OneQuery

        fallback = GeneratedQueries(queries_list=[OneQuery(query=state["question"])])
        return Command(
            goto="send_queries_in_parallel",
            update={"generated_questions": fallback},
        )


async def send_queries_in_parallel(
    state: Neo4jQueryState,
) -> Command[list[Send]]:
    """Node that fans out to search each generated question in parallel.

    This node creates Send commands for parallel execution of GraphRAG searches
    on all generated question variations.

    Args:
        state: The current graph state with generated questions.

    Returns:
        Command with Send objects for parallel GraphRAG searches.
    """
    lista_de_queries = [
        query.query_str for query in state["generated_questions"].queries_list
    ]

    logging.info(
        "Fanning out to search %d question variations in parallel",
        len(lista_de_queries),
    )
    logging.debug("Questions to search: %s", lista_de_queries)

    sends = [
        Send(
            "generate_answer",
            {"query": query},
        )
        for query in lista_de_queries
    ]
    return Command(goto=sends)


async def generate_answer(
    state: Neo4jQueryState,
) -> Command[Literal[END]]:
    """Node that performs a GraphRAG search for a single query.

    This node executes the GraphRAG search asynchronously and returns the answer.
    Error handling ensures that failures in individual searches don't crash
    the entire pipeline. Results from all parallel executions are automatically
    accumulated into a deduplicated list by the reduce_lists reducer.

    Args:
        state: The current graph state with a single query.

    Returns:
        Command updating results with the GraphRAG answer and ending execution.
        The answer is added to the accumulated results list via the reducer.
    """
    query_str = state["query"]
    logging.info("Executing GraphRAG search for: '%s'", query_str)

    try:
        # Use the async wrapper to avoid blocking
        response = await async_graph_rag_search(
            query_str,
            retriever_config={"top_k": 5},
            return_context=False,
        )

        answer = response.answer if hasattr(response, "answer") else str(response)
        logging.info("GraphRAG search completed successfully for: '%s'", query_str)
        logging.debug(
            "Answer preview: %s...", answer[:200] if len(answer) > 200 else answer
        )
        # Note: results will be accumulated by the reducer (reduce_lists)
        # Each parallel execution contributes its answer to the accumulated list

        return Command(goto=END, update={"results": [answer]})

    except Exception as exc:
        error_msg = f"GraphRAG search failed for '{query_str}': {exc}"
        logging.error(error_msg)
        # Return error message as result to maintain pipeline flow
        return Command(
            goto=END,
            update={"results": [json.dumps({"error": error_msg}, ensure_ascii=False)]},
        )
