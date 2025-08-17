"""This module builds the graph for the Cypher Query Agent.

It uses the agent_logic module to define the nodes and edges of the graph.

The graph is compiled and returned.

The graph is used to run the Cypher Query Agent.

uv run -m src.agents.cypher_query_agent.graph_builder

"""

# %%
import asyncio

from langgraph.graph import START, StateGraph

from src.agents.cypher_query_agent.agent_logic import (
    generate_answer,
    generate_cypher_queries_in_parallel,
    generate_cypher_query,
    generate_questions,
    run_cypher_query,
    run_cypher_query_in_parallel,
)
from src.agents.cypher_query_agent.schemas import Neo4jQueryState


builder = StateGraph(Neo4jQueryState)

# 1. Generate questions
builder.add_node("generate_questions", generate_questions)
builder.add_node(
    "generate_cypher_queries_in_parallel", generate_cypher_queries_in_parallel
)

# 2. Generate Cypher queries
builder.add_node("generate_cypher_query", generate_cypher_query)
builder.add_node("run_cypher_query_in_parallel", run_cypher_query_in_parallel)

# 3. Run Cypher queries and generate answer
builder.add_node("run_cypher_query", run_cypher_query)
builder.add_node("generate_answer", generate_answer)

# 4. Define the graph's entry
builder.add_edge(START, "generate_questions")

graph = builder.compile()

if __name__ == "__main__":

    async def _main() -> None:
        async for _chunk in graph.astream(
            {"question": "la comuna con más proyectos"},
            stream_mode="updates",
            subgraphs=True,
            debug=True,
        ):
            pass

    asyncio.run(_main())
