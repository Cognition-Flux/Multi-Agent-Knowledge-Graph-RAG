# %%
from __future__ import annotations

from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.graph import END, MessagesState
from langgraph.types import Command, Send
from pydantic import Field

from KnowledgeGraphDB.Neo4j_KG_creation.cypher_runner import run_cypher
from src.agents.cypher_query_agent.llm_chains import (
    get_cypher_query_chain,
    get_question_generation_chain,
)
from src.agents.cypher_query_agent.reducers import reduce_lists
from src.agents.cypher_query_agent.schemas import GeneratedQueries


load_dotenv(override=True)


def sanitise_query(query: str) -> str:
    """Sanitise the query in case the LLM returned it inside markdown fences."""
    if query.startswith("```"):
        # Remove leading/trailing code fences
        stripped = query.strip("`").strip()
        # If language identifier present (e.g. ```cypher), drop first line
        if "\n" in stripped:
            first_line, rest = stripped.split("\n", 1)
            query = rest if first_line.lower().startswith("cypher") else stripped
        else:
            query = stripped
    return query


def safe_run_cypher(query: str) -> str | list[dict[str, any]]:
    """Devuelve el resultado de la consulta o un string de error en formato de lista."""
    try:
        return run_cypher(query)
    except Exception as exc:
        return [f"ERROR: {exc}"]


class Neo4jQueryState(MessagesState):
    """State of the Neo4j Graph RAG."""

    question: str = Field(default_factory=lambda: "")
    generated_questions: GeneratedQueries = Field(
        default_factory=lambda: GeneratedQueries(queries_list=[])
    )
    query: str = Field(default_factory=lambda: "")
    cypher_query: str = Field(default_factory=lambda: "")
    cypher_queries: Annotated[list[str], reduce_lists] = Field(default_factory=list)
    results: Annotated[list[str], reduce_lists] = Field(default_factory=list)


cypher_chain = get_cypher_query_chain(group="FEW_SHOTS_CYPHER_QUERY", k=2)
qgen_chain = get_question_generation_chain(group="FEW_SHOTS_QUESTIONS_GENERATION", k=2)


# %%
async def generate_questions(
    state: Neo4jQueryState,
) -> Command[Literal["generate_cypher_query"]]:
    """Node that generates queries."""
    generated_questions = await qgen_chain.ainvoke({"input": state["question"]})
    return Command(
        goto="generate_cypher_queries_in_parallel",
        update={"generated_questions": generated_questions},
    )


async def generate_cypher_queries_in_parallel(
    state: Neo4jQueryState,
) -> Command[list[Send]]:
    """Node that generates Cypher queries in parallel."""
    lista_de_queries = [
        query.query_str for query in state["generated_questions"].queries_list
    ]

    print(f"lista_de_queries: {lista_de_queries}")
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
    """Node that generates a Cypher query."""
    query_str = state["query"]

    response = await cypher_chain.ainvoke({"input": query_str})
    raw_query = response.cypher_query.strip()

    cypher_query = sanitise_query(raw_query)

    return Command(
        goto="run_cypher_query_in_parallel", update={"cypher_queries": [cypher_query]}
    )


async def run_cypher_query_in_parallel(
    state: Neo4jQueryState,
) -> Command[list[Send]]:
    """Node that runs a Cypher query."""
    lista_de_cypher_queries = list(state["cypher_queries"])
    print(f"lista_de_cypher_queries: {lista_de_cypher_queries}")
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
    """Node that runs a Cypher query."""
    cypher_query = state["cypher_query"]
    print(f"################## cypher_query: {cypher_query}")
    results = str(safe_run_cypher(cypher_query))
    print(f"################## results: {results}")

    return Command(goto="generate_answer", update={"results": [results]})


async def generate_answer(
    state: Neo4jQueryState,
) -> Command[Literal[END]]:
    """Node that generates an answer."""
    results = state["results"]
    print(f"################## results: {results}")
    question = state["question"]
    print(f"################## question: {question}")

    input_for_llm = (
        f"La pregunta es: {question} y la información para responderla es: {results}"
    )
    response = await llm.ainvoke(input_for_llm)

    return Command(goto=END, update={"messages": [AIMessage(content=response.content)]})


# %%
