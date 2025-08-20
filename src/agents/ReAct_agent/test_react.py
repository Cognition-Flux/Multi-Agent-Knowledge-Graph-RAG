"""Test script for the ReAct agent with a hardcoded plan.

This allows testing the ReAct agent without requiring the planner chain.

Usage:
uv run -m src.agents.ReAct_agent.test_react
"""

import asyncio
import logging

from src.agents.planner_agent.schemas import Plan, Step
from src.agents.ReAct_agent.graph_builder import graph


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_with_simple_plan() -> None:
    """Test the ReAct agent with a simple hardcoded plan."""
    # Create a simple test plan
    test_plan = Plan(
        goal="Get information about projects and flora in Antofagasta region",
        steps=[
            Step(
                instruction="Find projects in the Antofagasta region",
                suggested_tool="cypher_query_agent",
                reasoning="Use metadata query to find projects",
                result="",
                is_complete=False,
            ),
            Step(
                instruction="Find important flora species in Antofagasta",
                suggested_tool="hybrid_graphRAG_agent",
                reasoning="Use content search to find flora information",
                result="",
                is_complete=False,
            ),
        ],
        direct_response_to_the_user="",
    )

    logger.info("🎯 Testing ReAct agent with hardcoded plan")
    logger.info(f"   Goal: {test_plan.goal}")
    logger.info(f"   Steps: {len(test_plan.steps)}")

    # Execute the plan
    final_state = None
    async for chunk in graph.astream(
        {"plan": test_plan},
        stream_mode="updates",
        debug=False,  # Set to True for detailed output
    ):
        if "finish" in chunk:
            final_state = chunk["finish"]

    # Print final answer
    if final_state and "final_answer" in final_state:
        logger.info("\n✅ Final Answer:")
        print(final_state["final_answer"])

    logger.info("\n✅ Test completed successfully")


async def test_with_direct_response() -> None:
    """Test the ReAct agent with a direct response (no steps)."""
    test_plan = Plan(
        goal="Simple greeting",
        steps=[],
        direct_response_to_the_user="Hello! This is a direct response that doesn't require any tools.",
    )

    logger.info("🎯 Testing ReAct agent with direct response")

    final_state = None
    async for chunk in graph.astream(
        {"plan": test_plan},
        stream_mode="updates",
        debug=False,
    ):
        if "finish" in chunk:
            final_state = chunk["finish"]

    if final_state and "final_answer" in final_state:
        logger.info("\n✅ Final Answer:")
        print(final_state["final_answer"])

    logger.info("\n✅ Direct response test completed")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing ReAct Agent")
    logger.info("=" * 60)

    # Test with a plan that has steps
    asyncio.run(test_with_simple_plan())

    print("\n" + "=" * 60 + "\n")

    # Test with direct response
    asyncio.run(test_with_direct_response())
