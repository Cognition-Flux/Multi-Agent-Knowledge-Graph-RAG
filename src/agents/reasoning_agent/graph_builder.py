"""Graph builder for the reasoning agent.

This module builds the LangGraph workflow for the reasoning agent that performs
intellectual processing tasks on results from other agents.

Usage:
uv run -m src.agents.reasoning_agent.graph_builder
"""

import asyncio
import logging

from langgraph.graph import START, StateGraph

from src.agents.reasoning_agent.agent_logic import (
    parse_instruction,
    reason,
    synthesize,
)
from src.agents.reasoning_agent.schemas import ReasoningState


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Build the reasoning graph
builder = StateGraph(ReasoningState)

# Add nodes
builder.add_node("parse_instruction", parse_instruction)
builder.add_node("reason", reason)
builder.add_node("synthesize", synthesize)

# Define the entry point - start by parsing the instruction
builder.add_edge(START, "parse_instruction")

# Compile the graph
graph = builder.compile()


# Test function
if __name__ == "__main__":

    async def test_summarization() -> None:
        """Test the reasoning agent with a summarization task."""
        test_input = {
            "instruction": "Summarize the key findings from these project results and identify the most important patterns",
            "current_results": [
                {
                    "project": "Project Alpha",
                    "status": "completed",
                    "budget": "$1.2M",
                    "timeline": "6 months",
                    "outcome": "Successfully delivered all features",
                },
                {
                    "project": "Project Beta",
                    "status": "in progress",
                    "budget": "$800K",
                    "timeline": "4 months",
                    "outcome": "On track, 75% complete",
                },
                {
                    "project": "Project Gamma",
                    "status": "planning",
                    "budget": "$2.5M",
                    "timeline": "12 months",
                    "outcome": "Requirements gathering phase",
                },
            ],
            "partial_results": [],
        }

        logger.info("🧪 Testing reasoning agent with summarization task")

        final_state = None
        async for chunk in graph.astream(
            test_input,
            stream_mode="updates",
            debug=False,
        ):
            if "synthesize" in chunk:
                final_state = chunk["synthesize"]

        if final_state and "final_output" in final_state:
            logger.info("\n📊 Final Output:")
            print(final_state["final_output"])

        logger.info("\n✅ Summarization test completed")

    async def test_analysis() -> None:
        """Test the reasoning agent with an analysis task."""
        test_input = {
            "instruction": "Analyze the flora data and explain the ecological significance of the findings",
            "current_results": [
                "Region has 979 vascular plant species",
                "259 species in Desierto Costero de Tocopilla",
                "Over 50% of species are endemic",
                "Coastal fog supports biodiversity",
            ],
            "partial_results": [
                "High endemism indicates ecological isolation",
                "Adaptation to arid conditions",
            ],
        }

        logger.info("🧪 Testing reasoning agent with analysis task")

        final_state = None
        async for chunk in graph.astream(
            test_input,
            stream_mode="updates",
            debug=False,
        ):
            if "synthesize" in chunk:
                final_state = chunk["synthesize"]

        if final_state and "final_output" in final_state:
            logger.info("\n🌿 Final Output:")
            print(final_state["final_output"])

        logger.info("\n✅ Analysis test completed")

    # Run tests
    logger.info("=" * 60)
    logger.info("Testing Reasoning Agent")
    logger.info("=" * 60)

    asyncio.run(test_summarization())

    print("\n" + "=" * 60 + "\n")

    asyncio.run(test_analysis())
