"""This module builds the graph for the Hybrid GraphRAG Agent.

It uses the agent_logic module to define the nodes and edges of the graph.

The graph is compiled and returned for use in the Hybrid GraphRAG Agent.

Usage:
uv run -m src.agents.hybrid_graphRAG_agent.graph_builder

"""

# %%
import asyncio

from langgraph.graph import START, StateGraph

from src.agents.cypher_query_agent.schemas import Neo4jQueryState
from src.agents.hybrid_graphRAG_agent.agent_logic import (
    generate_answer,
    generate_questions,
    send_queries_in_parallel,
)


# Build the graph
builder = StateGraph(Neo4jQueryState)

# Add nodes
builder.add_node("generate_questions", generate_questions)
builder.add_node("send_queries_in_parallel", send_queries_in_parallel)
builder.add_node("generate_answer", generate_answer)

# Define the graph's entry point
# All other routing is handled by Command objects in the nodes
builder.add_edge(START, "generate_questions")

# Compile the graph
graph = builder.compile()


if __name__ == "__main__":

    async def _main() -> None:
        """Test the hybrid GraphRAG agent with a sample question."""
        test_question = "¿Qué información tienes sobre proyectos de biosólidos?"

        print("🔍 Testing Hybrid GraphRAG Agent")
        print(f"📝 Question: {test_question}\n")

        async for chunk in graph.astream(
            {"question": test_question},
            stream_mode="updates",
            subgraphs=True,
            debug=True,
        ):
            # The streaming will show progress
            pass

        print("\n✅ Test completed")

    asyncio.run(_main())
