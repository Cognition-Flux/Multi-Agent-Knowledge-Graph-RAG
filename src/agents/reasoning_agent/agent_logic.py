"""Agent logic for the reasoning agent - nodes and control flow."""

from __future__ import annotations

import json
import logging
from typing import Literal

from langgraph.types import Command

from src.agents.reasoning_agent.llm_chains import (
    get_reasoning_engine_chain,
    get_synthesizer_chain,
    get_task_parser_chain,
)
from src.agents.reasoning_agent.schemas import ReasoningState


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def parse_instruction(state: ReasoningState) -> Command[Literal["reason"]]:
    """Parse the instruction to understand the reasoning task."""
    instruction = state["instruction"]
    current_results = state["current_results"]

    logger.info(f"📝 Parsing instruction: {instruction[:100]}...")

    # Convert results to string for the parser
    results_str = _format_results(current_results)

    try:
        parser_chain = get_task_parser_chain()
        task = await parser_chain.ainvoke(
            {"instruction": instruction, "current_results": results_str}
        )

        logger.info(f"   Task type: {task.task_type}")
        logger.info(f"   Focus: {task.focus}")

        return Command(goto="reason", update={"reasoning_task": task})

    except Exception as e:
        logger.error(f"❌ Error parsing instruction: {e}")
        # Create a default task
        from src.agents.reasoning_agent.schemas import ReasoningTask

        default_task = ReasoningTask(task_type="analyze", focus=instruction, context="")
        return Command(goto="reason", update={"reasoning_task": default_task})


async def reason(state: ReasoningState) -> Command[Literal["synthesize"]]:
    """Perform the reasoning task."""
    task = state["reasoning_task"]
    current_results = state["current_results"]
    partial_results = state["partial_results"]

    logger.info(f"🧠 Performing {task.task_type} reasoning...")

    # Format results for reasoning
    current_str = _format_results(current_results)
    partial_str = _format_results(partial_results)

    try:
        reasoning_chain = get_reasoning_engine_chain()
        response = await reasoning_chain.ainvoke(
            {
                "task_type": task.task_type,
                "focus": task.focus,
                "context": task.context,
                "current_results": current_str,
                "partial_results": partial_str,
            }
        )

        logger.info(f"   Confidence: {response.confidence:.2f}")
        logger.info(f"   Key points: {len(response.key_points)}")

        return Command(goto="synthesize", update={"reasoning_response": response})

    except Exception as e:
        logger.warning(
            f"Reasoning structured output failed; using fallback. Error: {e}"
        )
        # Create a fallback response
        from src.agents.reasoning_agent.schemas import ReasoningResponse

        fallback = ReasoningResponse(
            reasoning="Error occurred during reasoning",
            conclusion=str(e),
            confidence=0.1,
            key_points=[],
        )
        return Command(goto="synthesize", update={"reasoning_response": fallback})


async def synthesize(state: ReasoningState) -> Command:
    """Synthesize the final output from the reasoning."""
    instruction = state["instruction"]
    response = state["reasoning_response"]

    logger.info("📄 Synthesizing final output...")

    try:
        synthesizer_chain = get_synthesizer_chain()

        # Format key points
        key_points_str = "\n".join(f"- {point}" for point in response.key_points)

        output = await synthesizer_chain.ainvoke(
            {
                "instruction": instruction,
                "reasoning": response.reasoning,
                "conclusion": response.conclusion,
                "key_points": key_points_str,
            }
        )

        # Extract content from the response
        final_text = output.content if hasattr(output, "content") else str(output)

        logger.info(f"✅ Output generated: {len(final_text)} characters")

        return Command(update={"final_output": final_text})

    except Exception as e:
        logger.error(f"❌ Error synthesizing output: {e}")
        # Fallback to the conclusion
        return Command(update={"final_output": response.conclusion})


def _format_results(results: list) -> str:
    """Format results list into a readable string."""
    if not results:
        return "No results available"

    formatted = []
    for i, result in enumerate(results, 1):
        if isinstance(result, dict):
            # Try to format as JSON
            try:
                formatted.append(f"{i}. {json.dumps(result, indent=2)}")
            except:
                formatted.append(f"{i}. {result!s}")
        elif isinstance(result, str):
            # Truncate very long strings
            if len(result) > 1000:
                formatted.append(f"{i}. {result[:1000]}...")
            else:
                formatted.append(f"{i}. {result}")
        else:
            formatted.append(f"{i}. {result!s}")

    return "\n\n".join(formatted)
