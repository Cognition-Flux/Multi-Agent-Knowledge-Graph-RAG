"""FastAPI service that streams LangGraph chunks.

Run with e.g.:

    uvicorn KnowledgeGraphDB.graph_creation.API_for_graph:app --reload

Open a browser or use *curl* / *httpie*:

    curl -N "http://localhost:8000/graph?question=¿De+qué+región+es+el+proyecto?"

The response is an event-stream (one JSON line per chunk).
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Scalar integration (beautiful API docs)
try:
    from scalar_fastapi import get_scalar_api_reference, Layout  # type: ignore
except ImportError:  # pragma: no cover
    # The dependency is optional; if missing, advise installing later.
    get_scalar_api_reference = None  # type: ignore


from KnowledgeGraphDB.graph_creation.graph_streamer import stream_graph

app = FastAPI(title="LangGraph Streaming API")

# CORS: allow requests from any origin (use with care in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Scalar Docs endpoint -------------------------------------------------------
# ---------------------------------------------------------------------------
if get_scalar_api_reference is not None:

    @app.get("/scalar", include_in_schema=False, summary="Scalar API Docs")
    async def scalar_html():  # noqa: D401
        """Serve polished API reference powered by *Scalar* (if installed)."""
        return get_scalar_api_reference(
            openapi_url=app.openapi_url,
            title=app.title,
            layout=Layout.MODERN,
            dark_mode=True,
        )

else:
    import logging

    logging.getLogger(__name__).info(
        "scalar-fastapi is not installed; /scalar docs endpoint disabled."
    )


async def _json_line_stream(question: str) -> AsyncGenerator[bytes, None]:
    """Yield each graph chunk serialised as UTF-8 encoded JSON lines."""

    async for chunk in stream_graph(question):
        # Default serialisation; customise as needed (e.g. SSE framing).
        yield (json.dumps(chunk, default=str) + "\n").encode()


@app.get("/graph", response_class=StreamingResponse, summary="Stream LangGraph chunks")
async def graph_endpoint(
    question: str = Query(
        ...,  # required
        description="Pregunta en lenguaje natural",
        example="cual es la región del documento?",
    ),
) -> StreamingResponse:  # noqa: D401
    """Devuelve los *chunks* generados por LangGraph en streaming.

    Los clientes obtienen un flujo *line-delimited JSON*; cada línea corresponde
    a un *chunk* emitido por LangGraph.
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="'question' cannot be empty")

    generator = _json_line_stream(question)
    return StreamingResponse(generator, media_type="text/event-stream")


# Convenience entry-point --------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "KnowledgeGraphDB.graph_creation.API_for_graph:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
"""
   curl -N -G \
        -H "Accept: text/event-stream" \
        --data-urlencode "question=¿De qué región es el proyecto?" \
        http://localhost:8000/graph
"""
