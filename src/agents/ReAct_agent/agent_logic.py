"""Agent logic for the ReAct agent - nodes and control flow.

This module defines the nodes that execute plan steps using the appropriate tools.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.types import Command

# Import the subgraphs as tools
from src.agents.cypher_query_agent.graph_builder import (
    graph as cypher_query_graph,
)
from src.agents.hybrid_graphRAG_agent.graph_builder import (
    graph as hybrid_graphRAG_graph,
)
from src.agents.ReAct_agent.schemas import ReActState, ToolResult
from src.agents.reasoning_agent.graph_builder import (
    graph as reasoning_graph,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_plan(state: ReActState) -> Command[Literal["execute_step", "finish"]]:
    """Check if there's a direct response or if we need to execute steps."""
    plan = state["plan"]

    # If there's a direct response and no steps, return it immediately
    if plan.direct_response_to_the_user and not plan.steps:
        logger.info("📝 Direct response available, no steps to execute")
        return Command(
            goto="finish",
            update={
                "final_answer": plan.direct_response_to_the_user,
                "is_complete": True,
            },
        )

    # If there are steps to execute, start executing them
    if plan.steps:
        logger.info(f"🎯 Plan has {len(plan.steps)} steps to execute")
        return Command(goto="execute_step")

    # No plan available
    logger.warning("⚠️ No plan or steps available")
    return Command(
        goto="finish",
        update={
            "final_answer": "No plan was provided to execute.",
            "is_complete": True,
            "errors": ["No plan provided"],
        },
    )


async def execute_step(state: ReActState) -> Command[Literal["reflect", "finish"]]:
    """Execute the current step using the appropriate tool."""
    plan = state["plan"]
    current_index = state["current_step_index"]

    # Check if we've completed all steps
    if current_index >= len(plan.steps):
        logger.info("✅ All steps completed")
        return Command(goto="finish")

    step = plan.steps[current_index]
    logger.info(
        f"🔧 Executing step {current_index + 1}/{len(plan.steps)}: {step.instruction}"
    )
    logger.info(f"   Using tool: {step.suggested_tool}")

    try:
        # Determine which tool to use based on the step's suggested_tool field
        tool_name = step.suggested_tool.lower()
        result = None

        if "cypher" in tool_name or "metadata" in tool_name:
            # Use cypher_query_agent for metadata queries
            logger.info("   📊 Using Cypher Query Agent")
            response = await cypher_query_graph.ainvoke({"question": step.instruction})
            result = (
                response.get("messages", [])[-1].content
                if response.get("messages")
                else str(response)
            )

        elif "hybrid" in tool_name or "graphrag" in tool_name or "chunk" in tool_name:
            # Use hybrid_graphRAG_agent for content queries
            logger.info("   🔍 Using Hybrid GraphRAG Agent")
            response = await hybrid_graphRAG_graph.ainvoke(
                {"question": step.instruction}
            )
            result = (
                response.get("messages", [])[-1].content
                if response.get("messages")
                else str(response)
            )

        elif "reasoning" in tool_name or "reason" in tool_name or "think" in tool_name:
            # Use reasoning_agent for intellectual tasks
            logger.info("   🧠 Using Reasoning Agent")

            # Gather all previous results for context
            previous_results = []
            for i, prev_step in enumerate(plan.steps[:current_index]):
                if prev_step.is_complete and prev_step.result:
                    previous_results.append(
                        {
                            "step": i + 1,
                            "instruction": prev_step.instruction,
                            "result": prev_step.result,
                        }
                    )

            # Invoke reasoning agent
            response = await reasoning_graph.ainvoke(
                {
                    "instruction": step.instruction,
                    "current_results": previous_results,
                    "partial_results": [r.result for r in state["tool_results"]],
                }
            )

            # Extract the final output
            result = response.get("final_output", str(response))

        else:
            # Default to reasoning agent for unknown tools
            logger.warning(
                f"   ⚠️ Unknown tool '{step.suggested_tool}', defaulting to Reasoning Agent"
            )

            # Gather context
            previous_results = []
            for i, prev_step in enumerate(plan.steps[:current_index]):
                if prev_step.is_complete and prev_step.result:
                    previous_results.append(
                        {
                            "step": i + 1,
                            "instruction": prev_step.instruction,
                            "result": prev_step.result,
                        }
                    )

            response = await reasoning_graph.ainvoke(
                {
                    "instruction": step.instruction,
                    "current_results": previous_results,
                    "partial_results": [],
                }
            )
            result = response.get("final_output", str(response))

        # Create tool result
        tool_result = ToolResult(
            tool_name=step.suggested_tool,
            step_index=current_index,
            result=result,
            error=None,
        )

        # Update the step with the result
        updated_plan = plan.model_copy(deep=True)
        updated_plan.steps[current_index].result = str(result)
        updated_plan.steps[current_index].is_complete = True

        logger.info(f"   ✅ Step {current_index + 1} completed successfully")

        return Command(
            goto="reflect",
            update={
                "plan": updated_plan,
                "tool_results": [tool_result],
                "current_step_index": current_index + 1,
            },
        )

    except Exception as e:
        logger.error(f"   ❌ Error executing step {current_index + 1}: {e!s}")

        # Create error result
        tool_result = ToolResult(
            tool_name=step.suggested_tool,
            step_index=current_index,
            result=None,
            error=str(e),
        )

        # Mark step as complete with error
        updated_plan = plan.model_copy(deep=True)
        updated_plan.steps[current_index].result = f"Error: {e!s}"
        updated_plan.steps[current_index].is_complete = True

        return Command(
            goto="reflect",
            update={
                "plan": updated_plan,
                "tool_results": [tool_result],
                "current_step_index": current_index + 1,
                "errors": [f"Step {current_index + 1} failed: {e!s}"],
            },
        )


async def reflect(state: ReActState) -> Command[Literal["execute_step", "finish"]]:
    """Reflect on the execution and decide next action."""
    plan = state["plan"]
    current_index = state["current_step_index"]

    # Check if all steps are complete
    if current_index >= len(plan.steps):
        logger.info("🏁 All steps executed, moving to finish")
        return Command(goto="finish")

    # Check if we should continue despite errors
    errors = state.get("errors", [])
    if len(errors) > 3:  # Too many errors, stop execution
        logger.error("❌ Too many errors, stopping execution")
        return Command(goto="finish")

    # Continue to next step
    logger.info(f"➡️ Moving to step {current_index + 1}/{len(plan.steps)}")
    return Command(goto="execute_step")


async def finish(state: ReActState) -> Command:
    """Finalize the execution and prepare the final answer."""
    plan = state["plan"]
    tool_results = state["tool_results"]

    # If we already have a final answer, use it
    if state.get("final_answer"):
        logger.info("📄 Using existing final answer")
        return Command(update={"is_complete": True})

    # Compile results from all executed steps
    if tool_results:
        logger.info("📝 Compiling results from executed steps")

        # Build a comprehensive answer from all results
        answer_parts = []
        answer_parts.append(f"**Goal:** {plan.goal}\n")

        for i, step in enumerate(plan.steps):
            if step.is_complete:
                answer_parts.append(f"\n**Step {i + 1}: {step.instruction}**")
                if step.result:
                    answer_parts.append(f"Result: {step.result}")

        final_answer = "\n".join(answer_parts)
    else:
        # No results available
        final_answer = (
            plan.direct_response_to_the_user
            if plan.direct_response_to_the_user
            else "No results available."
        )

    logger.info("✅ Execution complete")

    return Command(update={"final_answer": final_answer, "is_complete": True})
