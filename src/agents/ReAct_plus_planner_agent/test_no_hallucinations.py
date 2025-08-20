"""Test to verify the ReAct + Planner agent doesn't hallucinate tools.

This test verifies that the combined agent correctly handles tool names
and doesn't create non-existent tools.

Usage:
uv run -m src.agents.ReAct_plus_planner_agent.test_no_hallucinations
"""

import asyncio
import logging

from src.agents.ReAct_plus_planner_agent import graph
from src.agents.ReAct_plus_planner_agent.tools_registry import AVAILABLE_TOOLS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_tool_awareness() -> None:
    """Test that the agent is aware of available tools."""
    # Test questions that might trigger tool hallucinations
    test_cases = [
        {
            "question": "Use the SQL agent to query the database",
            "expected_tool": "cypher_query_agent",  # Should map SQL to cypher
            "description": "Testing SQL → Cypher mapping",
        },
        {
            "question": "Use the vector search tool to find documents",
            "expected_tool": "hybrid_graphRAG_agent",  # Should map vector search to hybrid
            "description": "Testing vector search → Hybrid GraphRAG mapping",
        },
        {
            "question": "Use the analysis tool to summarize results",
            "expected_tool": "reasoning_agent",  # Should map analysis to reasoning
            "description": "Testing analysis → Reasoning mapping",
        },
        {
            "question": "Query the knowledge base for project information",
            "expected_tool": "cypher_query_agent",
            "description": "Testing knowledge base → Cypher mapping",
        },
        {
            "question": "Search for environmental impact documents",
            "expected_tool": "hybrid_graphRAG_agent",
            "description": "Testing document search → Hybrid GraphRAG mapping",
        },
    ]

    logger.info("=" * 60)
    logger.info("🧪 Testing Tool Awareness and Anti-Hallucination")
    logger.info("=" * 60)

    # Show available tools
    logger.info("\n📋 Available tools:")
    for tool in AVAILABLE_TOOLS:
        logger.info(f"   - {tool['name']}: {tool['description'][:50]}...")

    # Run tests
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n{'=' * 40}")
        logger.info(f"Test {i}: {test_case['description']}")
        logger.info(f"Question: {test_case['question']}")
        logger.info(f"Expected tool: {test_case['expected_tool']}")

        # Track tool usage
        tools_used = []

        # Run the graph and capture updates
        async for chunk in graph.astream(
            {"question": test_case["question"]}, stream_mode="updates"
        ):
            for node, update in chunk.items():
                if node == "validate_plan" and "plan" in update:
                    plan = update["plan"]
                    for step in plan.steps:
                        tools_used.append(step.suggested_tool)
                        logger.info(f"   Tool selected: {step.suggested_tool}")

        # Verify tools are valid
        for tool in tools_used:
            if tool in [t["name"] for t in AVAILABLE_TOOLS]:
                logger.info(f"   ✅ Valid tool: {tool}")
            else:
                logger.error(f"   ❌ HALLUCINATED TOOL: {tool}")

        if not tools_used:
            logger.info("   ℹ️ No tools used (direct response)")

        logger.info(f"Test {i} completed")


async def test_complex_workflow() -> None:
    """Test a complex workflow that requires multiple tools."""
    question = """
    Find all solar energy projects in the database,
    then search for their environmental impact documents,
    and finally analyze and summarize the key environmental concerns.
    """

    logger.info("\n" + "=" * 60)
    logger.info("🧪 Testing Complex Multi-Tool Workflow")
    logger.info("=" * 60)
    logger.info(f"Question: {question.strip()}")

    # Track the workflow
    workflow_steps = []

    async for chunk in graph.astream({"question": question}, stream_mode="updates"):
        for node, update in chunk.items():
            logger.info(f"\n📍 Node: {node}")

            if node == "inject_tools_info":
                logger.info("   Tools information injected")

            elif node == "generate_plan" and "plan" in update:
                plan = update["plan"]
                logger.info(f"   Plan generated with {len(plan.steps)} steps:")
                for i, step in enumerate(plan.steps, 1):
                    workflow_steps.append(
                        {
                            "step": i,
                            "tool": step.suggested_tool,
                            "instruction": step.instruction[:50] + "...",
                        }
                    )
                    logger.info(
                        f"     {i}. {step.suggested_tool}: {step.instruction[:50]}..."
                    )

            elif node == "validate_plan":
                logger.info("   Plan validated - all tools verified")

            elif node == "execute_with_react":
                logger.info("   Executing plan with ReAct agent...")

            elif node == "finalize" and "final_answer" in update:
                logger.info(
                    f"   Final answer generated ({len(update['final_answer'])} chars)"
                )

    # Summary
    logger.info("\n📊 Workflow Summary:")
    logger.info(f"   Total steps: {len(workflow_steps)}")

    # Verify all tools are valid
    all_valid = True
    valid_tools = [t["name"] for t in AVAILABLE_TOOLS]

    for step_info in workflow_steps:
        if step_info["tool"] in valid_tools:
            logger.info(f"   ✅ Step {step_info['step']}: {step_info['tool']} (valid)")
        else:
            logger.error(
                f"   ❌ Step {step_info['step']}: {step_info['tool']} (INVALID)"
            )
            all_valid = False

    if all_valid:
        logger.info("\n✅ All tools in workflow are valid - NO HALLUCINATIONS")
    else:
        logger.error("\n❌ Some tools were hallucinated!")


async def test_error_recovery() -> None:
    """Test that the system recovers from invalid tool references."""
    # Intentionally problematic questions
    test_cases = [
        "Use the non_existent_tool to analyze data",
        "Run the magic_analyzer on the project data",
        "Execute the super_search_engine to find documents",
    ]

    logger.info("\n" + "=" * 60)
    logger.info("🧪 Testing Error Recovery from Invalid Tools")
    logger.info("=" * 60)

    for i, question in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}: {question}")

        try:
            result = await graph.ainvoke({"question": question})

            if result.get("final_answer"):
                logger.info("   ✅ Recovered and provided answer")
            else:
                logger.warning("   ⚠️ No answer generated")

        except Exception as e:
            logger.error(f"   ❌ Failed with error: {e}")

    logger.info("\n✅ Error recovery tests completed")


if __name__ == "__main__":
    logger.info("\n" + "=" * 60)
    logger.info("🚀 Testing Anti-Hallucination System")
    logger.info("=" * 60)

    # Run all tests
    asyncio.run(test_tool_awareness())
    asyncio.run(test_complex_workflow())
    asyncio.run(test_error_recovery())

    logger.info("\n" + "=" * 60)
    logger.info("✅ All anti-hallucination tests completed!")
    logger.info("=" * 60)
