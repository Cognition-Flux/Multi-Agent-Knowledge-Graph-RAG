"""FastAPI service that streams LangGraph chunks.

uv run -m src.API.streaming_api

curl http://0.0.0.0:8000/graph \
  --request POST \
  --header 'Content-Type: application/json' \
  --data '{
  "question": "¿Que comunas están en los proyectos?"
}'
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field


# Scalar integration (beautiful API docs)
try:
    from scalar_fastapi import Layout, get_scalar_api_reference  # type: ignore
except ImportError:  # pragma: no cover
    # The dependency is optional; if missing, advise installing later.
    get_scalar_api_reference = None  # type: ignore


from src.graph_streamers.async_stream_updates import stream_graph


app = FastAPI(title="LangGraph Streaming API")

# ---------------------------------------------------------------------------
# Request models -------------------------------------------------------------
# ---------------------------------------------------------------------------


class GraphRequest(BaseModel):
    """JSON body for POST /graph requests."""

    question: str = Field(description="Pregunta en lenguaje natural")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "¿Que comunas están en los proyectos?",
                }
            ]
        }
    )


# CORS: allow requests from any origin (use with care in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/scalar", include_in_schema=False, summary="Scalar API Docs")
async def scalar_html():
    if get_scalar_api_reference is None:  # pragma: no cover
        raise HTTPException(status_code=404, detail="Scalar docs not installed")
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
        layout=Layout.MODERN,
        dark_mode=True,
    )


async def _json_line_stream(question: str) -> AsyncGenerator[bytes, None]:
    """Yield each graph chunk serialised as UTF-8 encoded JSON lines."""
    async for chunk in stream_graph(question):
        # Default serialisation; customise as needed (e.g. SSE framing).
        yield (json.dumps(chunk, default=str, ensure_ascii=False) + "\n").encode(
            "utf-8"
        )


@app.post(
    "/graph",
    response_class=StreamingResponse,
    summary="Stream LangGraph chunks (JSON body)",
)
async def graph_endpoint_post(
    payload: GraphRequest = Body(
        examples={
            "demo": {
                "summary": "Pregunta por comunas (prefilled)",
                "value": {"question": "¿Que comunas están en los proyectos?"},
            }
        }
    ),
) -> StreamingResponse:
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="'question' cannot be empty")

    generator = _json_line_stream(payload.question)
    return StreamingResponse(generator, media_type="text/event-stream; charset=utf-8")


# Convenience entry-point --------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "src.API.streaming_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
