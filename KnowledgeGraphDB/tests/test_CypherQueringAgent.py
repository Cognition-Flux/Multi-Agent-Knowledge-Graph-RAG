# %%
from __future__ import annotations

import asyncio
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, Send
from pydantic import BaseModel, Field

from dev.langgraph_basics.Neo4jGraphRAG.cypher_runner import run_cypher
from dev.langgraph_basics.Neo4jGraphRAG.llm_chains_cypher import (
    chain_for_cypher_query,
    chain_for_questions_generation,
)


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


def reduce_lists(
    existing: list[str] | None,
    new: list[str] | str | Literal["delete"] | None,
) -> list[str]:
    """Combine two lists of strings in a robust way.

    Behaviour
    ---------
    • If *new* is the literal ``"delete"`` → returns an empty list (reset).
    • If either argument is ``None`` → treats it as an empty list.
    • Accepts *new* as a single string or list of strings.
    • Ensures the returned list has **unique items preserving order**.
    """
    # Reset signal
    if new == "delete":
        return []

    # Normalise inputs
    if existing is None:
        existing = []

    if new is None:
        new_items: list[str] = []
    elif isinstance(new, str):
        new_items = [new]
    else:
        new_items = list(new)

    combined: list[str] = existing + new_items

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for item in combined:
        if item not in seen:
            seen.add(item)
            deduped.append(item)

    return deduped


class OneQuery(BaseModel):
    """One query."""

    query_str: str


class GeneratedQueries(BaseModel):
    """Generated queries."""

    queries_list: list[OneQuery]


class CypherQuery(BaseModel):
    """Cypher query agent."""

    cypher_query: str = Field(description="The Cypher query ready to be executed.")


class Neo4jQueryState(MessagesState):
    """State of the Neo4j Graph RAG."""

    question: str = Field(default_factory=lambda: "")
    generated_questions: GeneratedQueries = Field(
        default_factory=lambda: GeneratedQueries(queries_list=[])
    )
    query: str = Field(default_factory=lambda: "")
    # cypher_query: CypherQuery = Field(
    #     default_factory=lambda: CypherQuery(cypher_query="")
    # )
    cypher_query: str = Field(default_factory=lambda: "")
    cypher_queries: Annotated[list[str], reduce_lists] = Field(default_factory=list)
    results: Annotated[list[str], reduce_lists] = Field(default_factory=list)


async def generate_questions(
    state: Neo4jQueryState,
) -> Command[Literal["generate_cypher_query"]]:
    """Node that generates queries."""
    generated_questions = await chain_for_questions_generation.ainvoke(
        {"input": state["question"]}
    )
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

    # lista_de_queries = [
    #     "Todos los nombres de las enzimas",
    #     "Todos los nombres de los metabolitos",
    # ]
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

    response = await chain_for_cypher_query.ainvoke({"input": query_str})
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


llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1-mini",
    api_version=os.getenv("AZURE_API_VERSION"),
    temperature=0,
    max_tokens=None,
    timeout=1200,
    max_retries=5,
    streaming=True,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)


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


builder = StateGraph(Neo4jQueryState)

builder.add_node("generate_questions", generate_questions)
builder.add_node(
    "generate_cypher_queries_in_parallel", generate_cypher_queries_in_parallel
)
builder.add_node("generate_cypher_query", generate_cypher_query)
builder.add_node("run_cypher_query_in_parallel", run_cypher_query_in_parallel)
builder.add_node("run_cypher_query", run_cypher_query)
builder.add_node("generate_answer", generate_answer)
builder.add_edge(START, "generate_questions")
graph = builder.compile()
# %%
if __name__ == "__main__":

    async def _main() -> None:
        async for _chunk in graph.astream(
            {"question": "¿Cuáles son los proyectos que tienen anexos?"},
            stream_mode="updates",
            subgraphs=True,
            debug=True,
        ):
            pass

    asyncio.run(_main())
