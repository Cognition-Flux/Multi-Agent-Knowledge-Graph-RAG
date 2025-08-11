"""Async streamer for supervisor agent using updates mode.

This module provides an async generator that streams the final response
from the supervisor agent, which routes questions to either the Cypher
query agent or the hybrid GraphRAG agent.

Example (FastAPI):
------------------

>>> from fastapi import FastAPI
>>> from fastapi.responses import StreamingResponse
>>> from src.graph_streamers.async_stream_by_updates import stream_supervisor_updates
>>>
>>> app = FastAPI()
>>>
>>> @app.get("/ask")
>>> async def ask_endpoint(question: str):
...     # StreamingResponse will consume the async generator and stream
...     # the final supervisor response to the client
...     return StreamingResponse(
...         stream_supervisor_updates(question), media_type="text/event-stream"
...     )
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import AIMessage

# Import the compiled supervisor graph
from src.agents.supervisor_agent.agent_logic import supervisor_graph


async def stream_supervisor_updates(
    question: str,
    *,
    stream_mode: str = "updates",
    subgraphs: bool = True,
    debug: bool = False,
    recursion_limit: int = 50,
    **extra_state: str | int | float | dict[str, Any] | list[Any],
) -> AsyncGenerator[str, None]:
    """Stream ONLY the final response from the supervisor agent.

    This function streams the supervisor's routing decision and the final
    answer from the appropriate subgraph (Cypher or GraphRAG agent).

    Parameters
    ----------
    question : str
        Natural-language question that will be routed by the supervisor
        to the appropriate specialized agent.
    stream_mode : str, optional
        Streaming mode for graph.astream. Defaults to "updates" to stream
        only state modifications.
    subgraphs : bool, optional
        Whether to include subgraph updates in the stream. Defaults to True.
    debug : bool, optional
        Enable debug output for detailed execution trace. Defaults to False.
    recursion_limit : int, optional
        Maximum recursion depth for graph execution. Defaults to 50.
    **extra_state
        Additional state fields to merge with the initial state
        (e.g., user_id, session_id, context).

    Yields:
    ------
    str
        The final answer content from the supervisor agent as a string.
        Only the final response is yielded, not intermediate updates.

    Notes:
    -----
    The supervisor agent analyzes the question and routes it to:
    - cypher_agent: For metadata queries (counts, lists, locations)
    - graphrag_agent: For content queries (species, impacts, details)

    The function buffers all updates and only yields the final answer
    once the graph execution completes.
    """
    # Build initial state with question and any extra fields
    initial_state: dict[str, Any] = {
        "question": question,
        **extra_state,
    }

    # Buffer for the final answer
    final_answer: str | None = None

    # Stream updates from the supervisor graph
    async for chunk in supervisor_graph.astream(
        initial_state,
        stream_mode=stream_mode,
        subgraphs=subgraphs,
        debug=debug,
        config={"recursion_limit": recursion_limit},
    ):
        # Process the chunk based on its structure
        if isinstance(chunk, tuple) and len(chunk) >= 2:
            # Format: (node_name, updates) or (path, updates)
            updates = chunk[1]
        elif isinstance(chunk, dict):
            # Direct updates dictionary
            updates = chunk
        else:
            continue

        # Extract final answer from updates
        if isinstance(updates, dict):
            # Check for direct final_answer field
            if "final_answer" in updates:
                final_answer = updates["final_answer"]

            # Also check for messages in node updates
            for node_name, node_update in updates.items():
                if isinstance(node_update, dict):
                    # Check for final_answer in node update
                    if "final_answer" in node_update:
                        final_answer = node_update["final_answer"]

                    # Check for messages that might contain the answer
                    if "messages" in node_update:
                        messages = node_update["messages"]
                        if isinstance(messages, list) and messages:
                            for msg in messages:
                                # Handle AIMessage objects
                                if isinstance(msg, AIMessage):
                                    content = msg.content
                                    if content:
                                        final_answer = content
                                # Handle dict messages
                                elif isinstance(msg, dict) and "content" in msg:
                                    content = msg["content"]
                                    if content:
                                        final_answer = content

    # Yield the final answer once streaming completes
    if final_answer:
        yield final_answer
    else:
        yield "No answer was generated by the supervisor agent."


async def stream_supervisor_with_metadata(
    question: str,
    *,
    stream_mode: str = "updates",
    include_routing: bool = True,
    **kwargs: Any,
) -> AsyncGenerator[dict[str, Any], None]:
    """Stream supervisor updates with metadata about routing decisions.

    This variant provides more detailed information including which agent
    was selected and why, useful for debugging or displaying routing info
    in the UI.

    Parameters
    ----------
    question : str
        The question to route and answer.
    stream_mode : str, optional
        Streaming mode. Defaults to "updates".
    include_routing : bool, optional
        Whether to include routing decision details. Defaults to True.
    **kwargs
        Additional arguments passed to stream_supervisor_updates.

    Yields:
    ------
    dict[str, Any]
        Dictionary containing:
        - type: "routing" or "answer"
        - content: The actual content
        - metadata: Additional information (agent selected, reasoning, etc.)
    """
    initial_state = {"question": question}

    routing_info = {}
    final_answer = None

    async for chunk in supervisor_graph.astream(
        initial_state,
        stream_mode=stream_mode,
        config={"recursion_limit": kwargs.get("recursion_limit", 50)},
        debug=kwargs.get("debug", False),
    ):
        if isinstance(chunk, tuple) and len(chunk) >= 2:
            node_name, updates = chunk[0], chunk[1]
        elif isinstance(chunk, dict):
            updates = chunk
            node_name = None
        else:
            continue

        # Check if this is from the supervisor decision node
        if node_name == "supervisor_decision" and include_routing:
            # Extract routing decision if available
            if isinstance(updates, dict):
                for key, value in updates.items():
                    if isinstance(value, dict):
                        # Look for routing information in the update
                        if "agent" in value or "reasoning" in value:
                            routing_info.update(value)
                            yield {
                                "type": "routing",
                                "content": f"Routing to {value.get('agent', 'unknown')}",
                                "metadata": {
                                    "agent": value.get("agent"),
                                    "reasoning": value.get("reasoning"),
                                },
                            }

        # Extract final answer
        if isinstance(updates, dict):
            if "final_answer" in updates:
                final_answer = updates["final_answer"]

            for node_update in updates.values():
                if isinstance(node_update, dict):
                    if "final_answer" in node_update:
                        final_answer = node_update["final_answer"]

                    if "messages" in node_update:
                        messages = node_update["messages"]
                        if isinstance(messages, list) and messages:
                            last_msg = messages[-1]
                            if isinstance(last_msg, AIMessage):
                                final_answer = last_msg.content
                            elif isinstance(last_msg, dict) and "content" in last_msg:
                                final_answer = last_msg["content"]

    # Yield the final answer with metadata
    if final_answer:
        yield {
            "type": "answer",
            "content": final_answer,
            "metadata": routing_info if include_routing else {},
        }
    else:
        yield {
            "type": "error",
            "content": "No answer was generated",
            "metadata": {},
        }


# --------------------------------------------------------------------------- #
# Main execution for module testing
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import asyncio
    import sys

    async def demo():
        """Demo function to test the streamer when run as a module."""
        print("=" * 70)
        print("🚀 Supervisor Agent Async Streamer Demo")
        print("=" * 70)

        # Get question from command line or use default
        if len(sys.argv) > 1:
            question = " ".join(sys.argv[1:])
        else:
            # Default demo questions
            demo_questions = [
                "¿Cuántos proyectos hay en total?",
                "¿Qué especies de flora están en peligro?",
                "¿En qué regiones hay proyectos?",
            ]

            print("\n📝 Available demo questions:")
            for i, q in enumerate(demo_questions, 1):
                print(f"  {i}. {q}")

            print("\n💡 Tip: You can also run with a custom question:")
            print(
                "  uv run -m src.graph_streamers.async_stream_by_updates 'Your question here'"
            )

            # Use the first demo question
            question = demo_questions[0]
            print(f"\n🎯 Using demo question: {question}")

        print("-" * 70)

        try:
            # Test basic streaming
            print("\n📊 Testing basic streaming...")
            print("-" * 50)

            async for answer in stream_supervisor_updates(question):
                print("\n✅ Final Answer:")
                print("-" * 50)
                print(answer)
                print("-" * 50)

            # Also test metadata streaming
            print("\n📊 Testing streaming with metadata...")
            print("-" * 50)

            async for update in stream_supervisor_with_metadata(
                question,
                include_routing=True,
            ):
                if update["type"] == "routing":
                    print("\n🎯 Routing Decision:")
                    print(f"  Agent: {update['metadata'].get('agent', 'N/A')}")
                    print(f"  Reasoning: {update['metadata'].get('reasoning', 'N/A')}")
                elif update["type"] == "answer":
                    print("\n✅ Answer Preview (first 200 chars):")
                    print(f"  {update['content'][:200]}...")
                elif update["type"] == "error":
                    print(f"\n❌ Error: {update['content']}")

        except KeyboardInterrupt:
            print("\n\n⚠️ Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during demo: {e}")
            import traceback

            traceback.print_exc()

        print("\n" + "=" * 70)
        print("✅ Demo completed!")
        print("=" * 70)

    # Run the async demo
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
