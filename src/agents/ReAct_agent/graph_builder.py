"""Graph builder for the ReAct agent.

This module builds the LangGraph workflow for the ReAct agent that executes
plans using cypher_query_agent and hybrid_graphRAG_agent as tools.

Usage:
uv run -m src.agents.ReAct_agent.graph_builder
"""

import asyncio
import logging

from langgraph.graph import START, StateGraph

from src.agents.planner_agent.llm_chains import get_planner_chain
from src.agents.ReAct_agent.agent_logic import (
    check_plan,
    execute_step,
    finish,
    reflect,
)
from src.agents.ReAct_agent.schemas import ReActState


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Build the ReAct graph
builder = StateGraph(ReActState)

# Add nodes
builder.add_node("check_plan", check_plan)
builder.add_node("execute_step", execute_step)
builder.add_node("reflect", reflect)
builder.add_node("finish", finish)

# Define the entry point - start by checking the plan
builder.add_edge(START, "check_plan")

# Compile the graph
graph = builder.compile()


# Test function
if __name__ == "__main__":

    async def _main() -> None:
        """Test the ReAct agent with a sample plan."""
        # First, generate a plan using the planner chain
        logger.info("🎯 Generating plan using planner chain...")
        planner_chain = get_planner_chain()

        # Test question that requires both metadata and content
        test_question = "Dame un resumen de los proyectos en la región de Antofagasta y menciona las especies de flora más importantes"

        plan_response = await planner_chain.ainvoke({"input": test_question})

        logger.info("📋 Generated Plan:")
        logger.info(f"   Goal: {plan_response.goal}")
        logger.info(f"   Steps: {len(plan_response.steps)}")
        for i, step in enumerate(plan_response.steps):
            logger.info(
                f"     {i + 1}. {step.instruction} (tool: {step.suggested_tool})"
            )

        # Execute the plan with the ReAct agent
        logger.info("\n🚀 Executing plan with ReAct agent...")

        final_state = None
        async for chunk in graph.astream(
            {"plan": plan_response},
            stream_mode="updates",
            subgraphs=True,
            debug=True,
        ):
            # Store the final state
            if "finish" in chunk:
                final_state = chunk["finish"]

        # Print results
        if final_state and "final_answer" in final_state:
            logger.info("\n📄 Final Answer:")
            logger.info(final_state["final_answer"])

        logger.info("\n✅ Test completed")

    asyncio.run(_main())
