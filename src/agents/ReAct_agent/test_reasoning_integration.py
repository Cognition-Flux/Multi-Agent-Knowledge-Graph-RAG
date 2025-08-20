"""Test the integration of the reasoning agent with the ReAct agent.

Usage:
uv run -m src.agents.ReAct_agent.test_reasoning_integration
"""

import asyncio
import logging

from src.agents.planner_agent.schemas import Plan, Step
from src.agents.ReAct_agent.graph_builder import graph


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_reasoning_integration() -> None:
    """Test that the ReAct agent can use the reasoning agent."""
    # Create a plan that uses multiple tools including reasoning
    test_plan = Plan(
        goal="Analyze project data and provide insights",
        steps=[
            Step(
                instruction="Find all projects in the Antofagasta region",
                suggested_tool="cypher_query_agent",
                reasoning="Query metadata to get project list",
                result="",
                is_complete=False,
            ),
            Step(
                instruction="Get detailed information about flora in the region",
                suggested_tool="hybrid_graphRAG_agent",
                reasoning="Use content search for flora details",
                result="",
                is_complete=False,
            ),
            Step(
                instruction="Analyze and summarize the key findings from the projects and flora data, identifying patterns and ecological significance",
                suggested_tool="Reasoning_agent",
                reasoning="Use reasoning to synthesize insights from previous results",
                result="",
                is_complete=False,
            ),
        ],
        direct_response_to_the_user="",
    )

    logger.info("🎯 Testing ReAct agent with reasoning agent integration")
    logger.info(f"   Goal: {test_plan.goal}")
    logger.info(f"   Steps: {len(test_plan.steps)}")
    for i, step in enumerate(test_plan.steps):
        logger.info(f"     {i + 1}. Tool: {step.suggested_tool}")

    # Execute the plan
    final_state = None
    try:
        async for chunk in graph.astream(
            {"plan": test_plan},
            stream_mode="updates",
            debug=False,
        ):
            if "finish" in chunk:
                final_state = chunk["finish"]
    except Exception as e:
        logger.error(f"Error during execution: {e}")

    # Print final answer
    if final_state and "final_answer" in final_state:
        logger.info("\n✅ Final Answer:")
        print("-" * 60)
        print(final_state["final_answer"])
        print("-" * 60)

    logger.info("\n✅ Integration test completed")


async def test_reasoning_as_default() -> None:
    """Test that unknown tools default to reasoning agent."""
    test_plan = Plan(
        goal="Test unknown tool handling",
        steps=[
            Step(
                instruction="Generate a thoughtful reflection on the importance of environmental monitoring",
                suggested_tool="UnknownTool",  # This should default to reasoning
                reasoning="Test default tool handling",
                result="",
                is_complete=False,
            ),
        ],
        direct_response_to_the_user="",
    )

    logger.info("🎯 Testing default tool handling (should use reasoning agent)")

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

    logger.info("\n✅ Default tool test completed")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing Reasoning Agent Integration with ReAct")
    logger.info("=" * 60)

    # Test full integration with multiple tools
    asyncio.run(test_reasoning_integration())

    print("\n" + "=" * 60 + "\n")

    # Test default tool handling
    asyncio.run(test_reasoning_as_default())
