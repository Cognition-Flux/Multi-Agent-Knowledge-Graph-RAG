"""Utility to stream LangGraph results via an async generator.

This helper wraps ``graph.astream`` so that it can be directly used
in frameworks (e.g. FastAPI / Starlette) that expect an **asynchronous
iterator** / **async generator** returning chunks that will be forwarded
straight to the client.

The streamer is configured to work with ReAct_plus_planner_agent and
streams only the final answer when it becomes available, with proper
formatting for human readability.

Example (FastAPI):
------------------

>>> from fastapi import FastAPI
>>> from fastapi.responses import StreamingResponse
>>> from src.graph_streamers.async_stream_updates import async_stream_graph
>>>
>>> app = FastAPI()
>>>
>>> @app.get("/stream")
... async def stream_endpoint(question: str):
...     # ``StreamingResponse`` will consume the async generator returned
...     # by ``async_stream_graph`` and send each chunk over the wire as soon
...     # as it is yielded.
...     return StreamingResponse(
...         async_stream_graph(question), media_type="text/event-stream"
...     )


uv run -m src.graph_streamers.async_stream_updates

"""

# %%
from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from typing import Any

# Import the ReAct + Planner combined agent graph
from src.agents.ReAct_plus_planner_agent.graph_builder import graph


def clean_and_format_answer(content: str) -> str:
    """Clean and format the answer for human readability.

    This function processes the raw output from the ReAct agent and extracts
    only the human-readable content, removing internal data structures like
    dictionaries with 'messages', 'results', etc.

    Parameters
    ----------
    content : str
        The raw content from the agent that may contain internal data structures.

    Returns:
    -------
    str
        Cleaned and formatted content suitable for display to end users.
    """
    if not content:
        return ""

    # Pattern to detect dictionary-like structures with 'results' field
    dict_pattern = r"\{'messages':[^}]*'results':\s*\[(.*?)\]\}"

    # Check if the content contains raw dictionary output
    if "'messages':" in content and "'results':" in content:
        # Try to extract just the results content
        match = re.search(r"'results':\s*\[(.*?)\]", content, re.DOTALL)
        if match:
            results_content = match.group(1)
            # Clean up the extracted results
            # Remove quotes and clean up formatting
            results_content = results_content.replace("\\n", "\n")
            results_content = results_content.replace("\\'", "'")
            # Remove leading/trailing quotes from individual results
            results_content = re.sub(r"^'|'$", "", results_content.strip())
            results_content = re.sub(r"',\s*'", "\n\n", results_content)

            # If we successfully extracted results, use them
            if results_content and results_content != "[]":
                content = results_content

    # Check for "Result:" patterns and clean them up
    if "Result:" in content:
        # Replace dictionary outputs after "Result:" with just the actual result
        def clean_result(match):
            result_text = match.group(1)
            # Remove surrounding quotes
            result_text = result_text.strip().strip("'").strip('"')
            return f"Result: {result_text}"

        content = re.sub(
            r"Result:\s*\{[^}]*'results':\s*\[(.*?)\]\}",
            clean_result,
            content,
            flags=re.DOTALL,
        )

    # Clean up any remaining internal structure indicators
    content = re.sub(r"\{'messages':[^}]*\}", "", content)
    content = re.sub(r"GeneratedQueries\(queries_list=\[.*?\]\)", "", content)
    content = re.sub(r"OneQuery\(query_str='(.*?)'\)", r"\1", content)

    # Remove empty results indicators
    content = content.replace("'cypher_queries': []", "")
    content = content.replace("'messages': []", "")
    content = content.replace("'question': ", "")
    content = content.replace("'generated_questions': ", "")

    # Clean up excessive whitespace and newlines
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"[ \t]+", " ", content)

    # Final cleanup of quotes and formatting
    content = content.replace("\\'", "'")
    content = content.replace("\\n", "\n")

    # Remove leading/trailing whitespace
    content = content.strip()

    return content


# %%
async def async_stream_graph(
    question: str,
    *,
    stream_mode: str = "updates",
    subgraphs: bool = False,
    debug: bool = False,
    **extra_state: str | int | float | dict[str, Any] | list[Any],
) -> AsyncGenerator[tuple[str | None, str | None, str | None], None]:
    """Asynchronously yield intermediate results and final answer from the ReAct+Planner agent.

    Parameters
    ----------
    question:
        Natural-language question that will be injected into the LangGraph
        state under the ``"question"`` key.
    stream_mode:
        Mode accepted by ``graph.astream``.  Defaults to ``"updates"`` to get
        state changes as they occur.
    subgraphs, debug:
        Passed verbatim to :py:meth:`langgraph.Graph.astream` so callers can
        tweak the behaviour if required.
    **extra_state:
        Any additional keys that should be merged into the initial state fed to
        the graph (e.g. user ID, conversation context, thread_id…).  All keyword
        arguments provided here will be added to the dictionary that makes up
        the initial graph state.

    Yields:
    ------
    tuple[str | None, str | None, str | None]
        A tuple of (chunk, reasoning, plot) where:
        - chunk: The intermediate results or final answer content when available
        - reasoning: None (placeholder for future reasoning streaming)
        - plot: None (placeholder for future plot streaming)
    """
    # The initial state passed to LangGraph.  By default we inject the question
    # but callers can extend this with arbitrary extra keys via ``extra_state``.
    initial_state: dict[str, Any] = {"question": question, **extra_state}

    # Track what we've already sent to avoid duplicates
    final_answer_sent = False
    last_plan = None
    sent_tool_results = set()

    async for chunk in graph.astream(
        initial_state,
        stream_mode=stream_mode,
        subgraphs=subgraphs,
        debug=debug,
    ):
        # In "updates" mode, chunk is a dict with node name as key
        for node_name, state_update in chunk.items():
            if not isinstance(state_update, dict):
                continue

            # Check for plan generation (shows the goal and steps)
            if "plan" in state_update and state_update.get("plan"):
                plan = state_update["plan"]
                # Only send if it's a new/updated plan
                if plan != last_plan:
                    last_plan = plan
                    if hasattr(plan, "goal") and hasattr(plan, "steps"):
                        # Format the plan for display
                        plan_output = f"**Goal:** {plan.goal}\n\n"
                        for i, step in enumerate(plan.steps, 1):
                            step_text = f"**Step {i}: {step.instruction}**"
                            if hasattr(step, "result") and step.result:
                                # Clean the step result
                                cleaned_result = clean_and_format_answer(step.result)
                                step_text += f"\nResult: {cleaned_result}"
                            plan_output += step_text + "\n\n"

                        # Send the plan update
                        yield (plan_output.strip(), None, None)

            # Check for tool results (intermediate results from each step)
            if "tool_results" in state_update and state_update.get("tool_results"):
                tool_results = state_update["tool_results"]
                for result in tool_results:
                    # Create a unique key for this result to avoid duplicates
                    result_key = f"{result.tool_name}_{result.step_index}_{str(result.result)[:50]}"

                    if result_key not in sent_tool_results:
                        sent_tool_results.add(result_key)

                        # Format the tool result
                        if result.error:
                            result_text = f"**Step {result.step_index + 1} Error:** {result.error}"
                        else:
                            # Clean the result
                            cleaned_result = clean_and_format_answer(str(result.result))
                            result_text = f"**Step {result.step_index + 1}: {result.tool_name}**\nResult: {cleaned_result}"

                        # Send this intermediate result
                        yield (result_text, None, None)

            # Check for final answer
            if "final_answer" in state_update:
                final_answer = state_update.get("final_answer")

                # Only yield if we have actual content and haven't sent it yet
                if final_answer and not final_answer_sent:
                    final_answer_sent = True
                    # Clean and format the answer for human readability
                    cleaned_answer = clean_and_format_answer(final_answer)

                    # Check if the final answer is repeating the plan and steps
                    # If we already sent intermediate results, check if final answer contains the same content
                    if sent_tool_results and last_plan:
                        # Check if the final answer is just repeating what we already sent
                        if "**Goal:**" in cleaned_answer and "**Step" in cleaned_answer:
                            # The final answer is repeating the plan, skip it to avoid duplication
                            continue

                    # Add a separator before final answer if there were intermediate results
                    if sent_tool_results or last_plan:
                        cleaned_answer = "\n---\n\n**Summary:**\n" + cleaned_answer

                    # Yield in the format expected by pathway_front/state.py
                    # (chunk, reasoning, plot)
                    yield (cleaned_answer, None, None)


if __name__ == "__main__":

    async def _main() -> None:
        """Test the streaming with ReAct+Planner agent."""
        print("Testing async_stream_graph with ReAct+Planner agent...")
        print("-" * 60)

        test_questions = [
            "¿Qué es el cambio climático?",
            "¿Cuántos proyectos hay en la región de Antofagasta?",
        ]

        for test_question in test_questions:
            print(f"\n📝 Question: {test_question}")
            print("-" * 60)

            async for chunk, reasoning, plot in async_stream_graph(
                test_question,
                stream_mode="updates",
                subgraphs=False,
                debug=False,
            ):
                if chunk:
                    print(f"\n✅ Cleaned Final Answer:\n{chunk}")
                    print("-" * 40)
                    print(f"Length: {len(chunk)} characters")
                    # Check if internal structures were removed
                    if "'messages':" in chunk or "'results':" in chunk:
                        print("⚠️ Warning: Internal structures still present!")
                    else:
                        print("✓ Internal structures successfully removed")
                if reasoning:
                    print(f"\n💭 Reasoning: {reasoning}")
                if plot:
                    print("\n📊 Plot data received")

    import asyncio

    asyncio.run(_main())
