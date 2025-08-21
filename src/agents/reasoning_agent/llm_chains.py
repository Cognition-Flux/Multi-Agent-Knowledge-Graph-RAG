"""LLM chains for the reasoning agent.

uv run -m src.agents.reasoning_agent.llm_chains
"""

from pathlib import Path

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from src.agents.reasoning_agent.schemas import ReasoningResponse, ReasoningTask
from src.utils import get_llm


# Load prompts
_PROMPTS_PATH = Path(__file__).with_name("system_prompts.yaml")
with _PROMPTS_PATH.open(encoding="utf-8") as f:
    _data = yaml.safe_load(f) or {}
    _prompts = _data.get("LLM_CHAIN_SYSTEM_PROMPTS", {})
    SYSTEM_PROMPT_TASK_PARSER = _prompts.get("SYSTEM_PROMPT_TASK_PARSER", "").strip()
    SYSTEM_PROMPT_REASONING_ENGINE = _prompts.get(
        "SYSTEM_PROMPT_REASONING_ENGINE", ""
    ).strip()
    SYSTEM_PROMPT_SYNTHESIZER = _prompts.get("SYSTEM_PROMPT_SYNTHESIZER", "").strip()


def get_task_parser_chain(temperature: float = 0) -> Runnable:
    """Get the chain for parsing reasoning tasks from instructions."""
    llm = get_llm().bind(temperature=temperature)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_TASK_PARSER),
            (
                "human",
                "Instruction: {instruction}\n\nCurrent Results: {current_results}",
            ),
        ]
    )

    pipeline: Runnable = prompt | llm.with_structured_output(ReasoningTask)
    return pipeline.with_retry(stop_after_attempt=3)


def get_reasoning_engine_chain(temperature: float = 0.1) -> Runnable:
    """Get the main reasoning engine chain."""
    llm = get_llm().bind(temperature=temperature)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT_REASONING_ENGINE
                + "\n\nDebes devolver un objeto JSON que cumpla exactamente con el esquema:"
                + ' {{"reasoning": string, "conclusion": string, "confidence": number (0..1), "key_points": string[]}}.',
            ),
            (
                "human",
                """Task Type: {task_type}
Focus: {focus}
Context: {context}

Current Results:
{current_results}

Partial Results:
{partial_results}

Please perform the reasoning task and provide your analysis.

Return only the JSON object that matches the schema above. Ensure both 'reasoning' and 'conclusion' are present.""",
            ),
        ]
    )

    pipeline: Runnable = prompt | llm.with_structured_output(ReasoningResponse)
    return pipeline.with_retry(stop_after_attempt=3)


def get_synthesizer_chain(temperature: float = 0) -> Runnable:
    """Get the chain for synthesizing final output."""
    llm = get_llm().bind(temperature=temperature)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT_SYNTHESIZER),
            (
                "human",
                """Original Instruction: {instruction}

Reasoning Process: {reasoning}

Conclusion: {conclusion}

Key Points:
{key_points}

Generate the final output that directly addresses the instruction.""",
            ),
        ]
    )

    return prompt | llm


if __name__ == "__main__":
    import asyncio

    async def test_chains():
        # Test task parser
        parser = get_task_parser_chain()
        task = await parser.ainvoke(
            {
                "instruction": "Summarize the key findings from the project data",
                "current_results": "Project A: 50% complete, Project B: 80% complete",
            }
        )
        print("Task parsed:", task.model_dump_json(indent=2))

        # Test reasoning engine
        engine = get_reasoning_engine_chain()
        response = await engine.ainvoke(
            {
                "task_type": task.task_type,
                "focus": task.focus,
                "context": task.context,
                "current_results": "Project A: 50% complete, Project B: 80% complete",
                "partial_results": "",
            }
        )
        print("\nReasoning response:", response.model_dump_json(indent=2))

        # Test synthesizer
        synthesizer = get_synthesizer_chain()
        output = await synthesizer.ainvoke(
            {
                "instruction": "Summarize the key findings",
                "reasoning": response.reasoning,
                "conclusion": response.conclusion,
                "key_points": "\n".join(f"- {point}" for point in response.key_points),
            }
        )
        print("\nFinal output:", output.content)

    asyncio.run(test_chains())
