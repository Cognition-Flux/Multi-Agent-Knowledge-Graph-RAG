"""Utility to stream LangGraph results via an async generator.

This helper wraps ``graph.astream`` so that it can be directly used
in frameworks (e.g. FastAPI / Starlette) that expect an **asynchronous
iterator** / **async generator** returning chunks that will be forwarded
straight to the client.

Example (FastAPI):
------------------

>>> from fastapi import FastAPI
>>> from fastapi.responses import StreamingResponse
>>> from KnowledgeGraphDB.graph_creation.graph_streamer import stream_graph
>>>
>>> app = FastAPI()
>>>
>>> @app.get("/stream")
... async def stream_endpoint(question: str):
...     # ``StreamingResponse`` will consume the async generator returned
...     # by ``stream_graph`` and send each chunk over the wire as soon
...     # as it is yielded.
...     return StreamingResponse(stream_graph(question), media_type="text/event-stream")


uv run -m src.graph_streamers.async_stream_updates

"""

# %%
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from langchain_core.messages import AIMessage

# Re-use the graph that is already built in the testing module.  When this grows
# into a proper library you may want to move the graph construction into a
# dedicated module, but for now importing it avoids duplicate logic.
from src.agents.cypher_query_agent.graph_builder import graph


# %%
async def async_stream_graph(
    question: str,
    *,
    stream_mode: str = "values",
    subgraphs: bool = True,
    debug: bool = True,
    **extra_state: str | int | float | dict[str, Any] | list[Any],
) -> AsyncGenerator[Any, None]:
    """Asynchronously yield ONLY the final message content produced by the LangGraph ``graph``.

    Parameters
    ----------
    question:
        Natural-language question that will be injected into the LangGraph
        state under the ``"question"`` key.
    stream_mode:
        Mode accepted by ``graph.astream``.  Defaults to ``"values"`` so that
        only modified parts of the state are sent downstream.
    subgraphs, debug:
        Passed verbatim to :py:meth:`langgraph.Graph.astream` so callers can
        tweak the behaviour if required.
    **extra_state:
        Any additional keys that should be merged into the initial state fed to
        the graph (e.g. user ID, conversation context…).  All keyword arguments
        provided here will be added to the dictionary that makes up the initial
        graph state.

    Yields:
    ------
    Any
        The raw *chunk* objects emitted by ``graph.astream``.  It is up to the
        consumer to decide how to serialise them (JSON, SSE, etc.).
    """
    # The initial state passed to LangGraph.  By default we inject the question
    # but callers can extend this with arbitrary extra keys via ``extra_state``.
    initial_state: dict[str, Any] = {"question": question, **extra_state}

    # Consume the async stream but buffer the last AI message content only.

    async for chunk in graph.astream(
        initial_state,
        stream_mode=stream_mode,
        subgraphs=subgraphs,
        debug=debug,
    ):
        _, messages = chunk
        if (
            messages.get("messages")
            and isinstance(messages["messages"], list)
            and len(messages["messages"]) > 0
            and isinstance(messages["messages"][-1], AIMessage)
        ):
            yield (messages["messages"][-1].content, None, None)


if __name__ == "__main__":

    async def _main() -> None:
        async for chunk in async_stream_graph(
            "total de tipologías por región",
            stream_mode="values",
            subgraphs=True,
            debug=False,
        ):
            print(chunk)

    import asyncio

    asyncio.run(_main())
