# %%
"""Quick manual test for the ``stream_graph`` async generator.

Run this module directly:

    python -m KnowledgeGraphDB.graph_creation.test_Streamer

It will issue a sample question to the LangGraph pipeline and print every
chunk as soon as it is streamed back.
"""

from __future__ import annotations

import asyncio

# from KnowledgeGraphDB.graph_creation.graph_streamer import stream_graph
from KnowledgeGraphDB.Neo4j_KG_creation.graph_streamer import stream_graph


async def main() -> None:
    """Send a sample question and display streamed chunks."""
    async for chunk in stream_graph(
        "¿De qué región es el proyecto?",
        stream_mode="updates",
        subgraphs=True,
        debug=True,
    ):
        print(chunk)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
