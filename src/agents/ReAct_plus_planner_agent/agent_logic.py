"""Agent logic for the combined ReAct + Planner agent.

This module defines the nodes that orchestrate the planning and execution
workflow with full tool awareness.
"""

import logging
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.types import Command

from src.agents.planner_agent.llm_chains import get_planner_chain
from src.agents.ReAct_agent import graph as react_graph
from src.agents.ReAct_plus_planner_agent.schemas import ReActPlusPlannerState
from src.agents.ReAct_plus_planner_agent.tools_registry import (
    format_tools_for_prompt,
    get_all_tool_names,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def inject_tools_info(
    state: ReActPlusPlannerState,
) -> Command[Literal["generate_plan"]]:
    """Inject tools information into the state for awareness."""
    logger.info("💉 Injecting tools information")

    # Get formatted tools information
    tools_info = format_tools_for_prompt()

    logger.info("   Available tools injected into context")

    return Command(goto="generate_plan", update={"available_tools": tools_info})


async def generate_plan(
    state: ReActPlusPlannerState,
) -> Command[Literal["validate_plan", "direct_answer"]]:
    """Generate a plan using the planner with tool awareness."""
    question = state["question"]
    tools_info = state["available_tools"]

    logger.info(f"📝 Generating plan for: {question[:100]}...")

    try:
        # Create an enhanced prompt with tools information
        enhanced_prompt = f"""
{tools_info}

User Question: {question}

Create a plan to answer this question using ONLY the available tools listed above.
For each step, specify the exact tool name from the list.
"""

        # Generate the plan
        planner_chain = get_planner_chain()
        plan = await planner_chain.ainvoke({"input": enhanced_prompt})

        logger.info(f"   Generated plan with {len(plan.steps)} steps")

        # Log the plan details
        for i, step in enumerate(plan.steps):
            logger.info(
                f"     Step {i + 1}: {step.suggested_tool} - {step.instruction[:50]}..."
            )

        # Check if there's a direct response
        if plan.direct_response_to_the_user and not plan.steps:
            logger.info("   📄 Direct response available, no execution needed")
            return Command(
                goto="direct_answer",
                update={"plan": plan, "final_answer": plan.direct_response_to_the_user},
            )

        return Command(goto="validate_plan", update={"plan": plan})

    except Exception as e:
        logger.error(f"❌ Error generating plan: {e}")
        return Command(
            goto="direct_answer",
            update={
                "errors": [f"Failed to generate plan: {e!s}"],
                "final_answer": "I encountered an error while planning how to answer your question.",
            },
        )


async def validate_plan(
    state: ReActPlusPlannerState,
) -> Command[Literal["execute_with_react"]]:
    """Validate that the plan uses only available tools."""
    plan = state["plan"]

    logger.info("✅ Validating plan tools")

    # List of valid tool names (core + structured tools)
    valid_tools = [
        "cypher_query_agent",
        "hybrid_graphRAG_agent",
        "reasoning_agent",
        *get_all_tool_names(),
    ]

    # Validate each step's tool
    for i, step in enumerate(plan.steps):
        tool_name = step.suggested_tool.lower()

        # Check if tool is valid or can be mapped
        if tool_name not in [t.lower() for t in valid_tools]:
            # Try to map to a valid tool
            if "reasoning" in tool_name or "reason" in tool_name:
                step.suggested_tool = "reasoning_agent"
                logger.warning(
                    f"   Step {i + 1}: Mapped '{step.suggested_tool}' to 'reasoning_agent'"
                )
            elif "cypher" in tool_name or "metadata" in tool_name:
                step.suggested_tool = "cypher_query_agent"
                logger.warning(
                    f"   Step {i + 1}: Mapped '{step.suggested_tool}' to 'cypher_query_agent'"
                )
            elif "hybrid" in tool_name or "graphrag" in tool_name:
                step.suggested_tool = "hybrid_graphRAG_agent"
                logger.warning(
                    f"   Step {i + 1}: Mapped '{step.suggested_tool}' to 'hybrid_graphRAG_agent'"
                )
            else:
                # Default to reasoning_agent for unknown tools
                logger.warning(
                    f"   Step {i + 1}: Unknown tool '{step.suggested_tool}', defaulting to 'reasoning_agent'"
                )
                step.suggested_tool = "reasoning_agent"
        else:
            logger.info(f"   Step {i + 1}: Tool '{step.suggested_tool}' is valid")

    return Command(goto="execute_with_react", update={"plan": plan})


async def execute_with_react(
    state: ReActPlusPlannerState,
) -> Command[Literal["finalize"]]:
    """Execute the plan using the ReAct agent."""
    plan = state["plan"]

    logger.info("🚀 Executing plan with ReAct agent")

    try:
        # Invoke the ReAct agent with the plan
        result = await react_graph.ainvoke({"plan": plan})

        # Extract results
        final_answer = result.get("final_answer", "")
        tool_results = result.get("tool_results", [])
        errors = result.get("errors", [])

        logger.info(f"   Execution completed with {len(tool_results)} tool calls")

        return Command(
            goto="finalize",
            update={
                "final_answer": final_answer,
                "tool_results": tool_results,
                "errors": errors,
            },
        )

    except Exception as e:
        logger.error(f"❌ Error during execution: {e}")
        return Command(
            goto="finalize",
            update={
                "errors": [f"Execution error: {e!s}"],
                "final_answer": "I encountered an error while executing the plan.",
            },
        )


async def direct_answer(state: ReActPlusPlannerState) -> Command[Literal["finalize"]]:
    """Handle direct answers that don't require execution."""
    logger.info("📄 Providing direct answer")

    # Add a message to indicate this was a direct response
    message = AIMessage(content=state["final_answer"])

    return Command(goto="finalize", update={"messages": [message], "is_complete": True})


async def finalize(state: ReActPlusPlannerState) -> Command:
    """Finalize the workflow and prepare the response."""
    final_answer = state.get("final_answer", "")
    errors = state.get("errors", [])

    if errors:
        logger.warning(f"⚠️ Completed with {len(errors)} errors")
        for error in errors:
            logger.warning(f"   - {error}")

    if not final_answer and state.get("tool_results"):
        # Try to compile an answer from tool results
        logger.info("📝 Compiling answer from tool results")
        results = []
        for result in state["tool_results"]:
            if result.result and not result.error:
                results.append(str(result.result))
        final_answer = "\n\n".join(results) if results else "No results available."

    # Create final message
    if final_answer:
        message = AIMessage(content=final_answer)
        logger.info("✅ Workflow completed successfully")
    else:
        message = AIMessage(content="I was unable to generate an answer.")
        logger.error("❌ No answer generated")

    return Command(
        update={
            "messages": [message],
            "final_answer": final_answer,
            "is_complete": True,
        }
    )
