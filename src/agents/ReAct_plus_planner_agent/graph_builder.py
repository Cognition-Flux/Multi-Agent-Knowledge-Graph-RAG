"""Graph builder for the combined ReAct + Planner agent.

This module builds the complete workflow that combines planning
and execution with full tool awareness.

Usage:
uv run -m src.agents.ReAct_plus_planner_agent.graph_builder
"""

import asyncio
import logging

from langgraph.graph import START, StateGraph

from src.agents.ReAct_plus_planner_agent.agent_logic import (
    direct_answer,
    execute_with_react,
    finalize,
    generate_plan,
    inject_tools_info,
    validate_plan,
)
from src.agents.ReAct_plus_planner_agent.schemas import ReActPlusPlannerState


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Build the combined graph
builder = StateGraph(ReActPlusPlannerState)

# Add nodes
builder.add_node("inject_tools_info", inject_tools_info)
builder.add_node("generate_plan", generate_plan)
builder.add_node("validate_plan", validate_plan)
builder.add_node("execute_with_react", execute_with_react)
builder.add_node("direct_answer", direct_answer)
builder.add_node("finalize", finalize)

# Define the entry point - always start by injecting tools info
builder.add_edge(START, "inject_tools_info")

# Compile the graph
graph = builder.compile()


# Test function
if __name__ == "__main__":
    # async def test_with_metadata_query() -> None:
    #     """Test with a question that requires metadata query."""
    #     question = "¿Cuántos proyectos hay en la región de Antofagasta?"

    #     logger.info("=" * 60)
    #     logger.info("🧪 Test 1: Metadata Query")
    #     logger.info(f"Question: {question}")
    #     logger.info("=" * 60)

    #     result = await graph.ainvoke({"question": question})

    #     if result.get("final_answer"):
    #         logger.info("\n📊 Final Answer:")
    #         print(result["final_answer"])

    #     logger.info("\n✅ Test 1 completed")

    # async def test_with_complex_query() -> None:
    #     """Test with a complex question requiring multiple tools."""
    #     question = """
    #     Dame un resumen de los proyectos en la región de Antofagasta
    #     y analiza las especies de flora más importantes encontradas.
    #     """

    #     logger.info("=" * 60)
    #     logger.info("🧪 Test 2: Complex Multi-Tool Query")
    #     logger.info(f"Question: {question.strip()}")
    #     logger.info("=" * 60)

    #     result = await graph.ainvoke({"question": question})

    #     if result.get("final_answer"):
    #         logger.info("\n📊 Final Answer:")
    #         print(result["final_answer"])

    #     logger.info("\n✅ Test 2 completed")

    # async def test_with_direct_response() -> None:
    #     """Test with a simple question that might get a direct response."""
    #     question = "¿Qué es el cambio climático?"

    #     logger.info("=" * 60)
    #     logger.info("🧪 Test 3: Direct Response Query")
    #     logger.info(f"Question: {question}")
    #     logger.info("=" * 60)

    #     result = await graph.ainvoke({"question": question})

    #     if result.get("final_answer"):
    #         logger.info("\n📊 Final Answer:")
    #         print(result["final_answer"])

    #     logger.info("\n✅ Test 3 completed")

    async def test_streaming() -> None:
        """Test streaming execution to see the workflow progress."""
        question = "Lista los proyectos de energía solar en región de Coquimbo"

        logger.info("=" * 60)
        logger.info("🧪 Test 4: Streaming Execution")
        logger.info(f"Question: {question}")
        logger.info("=" * 60)

        async for chunk in graph.astream(
            {"question": question}, stream_mode="updates", debug=True
        ):
            # Log each update
            for node, update in chunk.items():
                logger.info(f"📍 Node: {node}")
                if update.get("plan"):
                    plan = update["plan"]
                    logger.info(f"   Plan generated with {len(plan.steps)} steps")
                if "final_answer" in update:
                    logger.info(f"   Answer ready: {len(update['final_answer'])} chars")

        logger.info("\n✅ Test 4 completed")

    # Run tests
    logger.info("\n" + "=" * 60)
    logger.info("🚀 Testing ReAct + Planner Combined Agent")
    logger.info("=" * 60 + "\n")

    # # Run each test
    # asyncio.run(test_with_metadata_query())
    # print("\n" + "-" * 60 + "\n")

    # asyncio.run(test_with_complex_query())
    # print("\n" + "-" * 60 + "\n")

    # asyncio.run(test_with_direct_response())
    # print("\n" + "-" * 60 + "\n")

    asyncio.run(test_streaming())

    logger.info("\n" + "=" * 60)
    logger.info("✅ All tests completed!")
    logger.info("=" * 60)
