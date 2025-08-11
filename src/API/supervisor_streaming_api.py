"""FastAPI service that streams supervisor agent responses.

This API provides endpoints for streaming the final responses from the supervisor
agent, which intelligently routes questions to the appropriate specialized agent.

Run the API:
    uv run -m src.API.supervisor_streaming_api

Test with curl:
    curl http://0.0.0.0:8000/ask \
      --request POST \
      --header 'Content-Type: application/json' \
      --data '{
        "question": "¿Cuántos proyectos hay en total?"
      }'

    curl http://0.0.0.0:8000/ask-with-metadata \
      --request POST \
      --header 'Content-Type: application/json' \
      --data '{
        "question": "¿Qué especies de flora están en peligro?",
        "include_metadata": true
      }'
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field


# Scalar integration for beautiful API docs (optional)
try:
    from scalar_fastapi import Layout, get_scalar_api_reference  # type: ignore
except ImportError:  # pragma: no cover
    # The dependency is optional; if missing, we can still run
    get_scalar_api_reference = None  # type: ignore

from src.graph_streamers.async_stream_by_updates import (
    stream_supervisor_updates,
    stream_supervisor_with_metadata,
)


# ---------------------------------------------------------------------------
# FastAPI Application Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Supervisor Agent Streaming API",
    description="Stream responses from the supervisor agent that routes questions to specialized agents",
    version="1.0.0",
)

# CORS: allow requests from any origin (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class SupervisorRequest(BaseModel):
    """JSON body for supervisor agent requests."""

    question: str = Field(description="Natural language question in Spanish or English")
    include_metadata: bool = Field(
        default=False,
        description="Include routing decision and reasoning in response",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug output in the agent execution",
    )
    recursion_limit: int = Field(
        default=50,
        description="Maximum recursion depth for graph execution",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "question": "¿Cuántos proyectos hay en total?",
                    "include_metadata": False,
                },
                {
                    "question": "¿Qué especies de flora están en peligro?",
                    "include_metadata": True,
                },
                {
                    "question": "¿En qué regiones hay proyectos?",
                    "include_metadata": False,
                },
            ]
        }
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    service: str = Field(description="Service name")
    version: str = Field(description="API version")


# ---------------------------------------------------------------------------
# Streaming Generators
# ---------------------------------------------------------------------------


async def _stream_supervisor_response(
    question: str,
    debug: bool = False,
    recursion_limit: int = 50,
) -> AsyncGenerator[bytes, None]:
    """Stream the supervisor's final response as JSON lines."""
    async for answer in stream_supervisor_updates(
        question=question,
        debug=debug,
        recursion_limit=recursion_limit,
    ):
        # Wrap answer in JSON structure and encode as UTF-8
        response = {"type": "answer", "content": answer}
        yield (json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8")


async def _stream_supervisor_with_metadata(
    question: str,
    include_metadata: bool = True,
    debug: bool = False,
    recursion_limit: int = 50,
) -> AsyncGenerator[bytes, None]:
    """Stream supervisor response with optional routing metadata."""
    async for update in stream_supervisor_with_metadata(
        question=question,
        include_routing=include_metadata,
        debug=debug,
        recursion_limit=recursion_limit,
    ):
        # Each update is already a structured dict
        yield (json.dumps(update, ensure_ascii=False) + "\n").encode("utf-8")


async def _stream_sse_format(
    question: str,
    include_metadata: bool = False,
    debug: bool = False,
    recursion_limit: int = 50,
) -> AsyncGenerator[bytes, None]:
    """Stream in Server-Sent Events (SSE) format for better browser compatibility."""
    if include_metadata:
        async for update in stream_supervisor_with_metadata(
            question=question,
            include_routing=True,
            debug=debug,
            recursion_limit=recursion_limit,
        ):
            # Format as SSE with "data:" prefix
            sse_data = f"data: {json.dumps(update, ensure_ascii=False)}\n\n"
            yield sse_data.encode("utf-8")
    else:
        async for answer in stream_supervisor_updates(
            question=question,
            debug=debug,
            recursion_limit=recursion_limit,
        ):
            # Simple SSE format with just the answer
            response = {"type": "answer", "content": answer}
            sse_data = f"data: {json.dumps(response, ensure_ascii=False)}\n\n"
            yield sse_data.encode("utf-8")


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if the API service is healthy and running."""
    return HealthResponse(
        status="healthy",
        service="supervisor-streaming-api",
        version="1.0.0",
    )


@app.get("/scalar", include_in_schema=False, summary="Scalar API Documentation")
async def scalar_html():
    """Serve the Scalar API documentation interface."""
    if get_scalar_api_reference is None:  # pragma: no cover
        raise HTTPException(
            status_code=404,
            detail="Scalar docs not installed. Run: uv add scalar-fastapi",
        )
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
        layout=Layout.MODERN,
        dark_mode=True,
    )


@app.post(
    "/ask",
    response_class=StreamingResponse,
    summary="Stream supervisor agent response",
    tags=["Streaming"],
)
async def ask_supervisor(
    payload: SupervisorRequest = Body(
        examples={
            "metadata_query": {
                "summary": "Count projects (routes to Cypher agent)",
                "value": {
                    "question": "¿Cuántos proyectos hay en total?",
                    "include_metadata": False,
                },
            },
            "content_query": {
                "summary": "Species information (routes to GraphRAG agent)",
                "value": {
                    "question": "¿Qué especies de flora están en peligro?",
                    "include_metadata": True,
                },
            },
            "location_query": {
                "summary": "Regional projects (routes to Cypher agent)",
                "value": {
                    "question": "¿En qué regiones hay proyectos?",
                    "include_metadata": False,
                },
            },
        }
    ),
) -> StreamingResponse:
    """Stream the supervisor agent's response to a question.

    The supervisor analyzes the question and routes it to:
    - **Cypher Agent**: For metadata queries (counts, locations, project names)
    - **GraphRAG Agent**: For content queries (species, environmental impacts, details)

    Returns a streaming response in JSON lines format.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    generator = _stream_supervisor_response(
        question=payload.question,
        debug=payload.debug,
        recursion_limit=payload.recursion_limit,
    )
    return StreamingResponse(
        generator,
        media_type="application/x-ndjson; charset=utf-8",
        headers={"X-Accel-Buffering": "no"},  # Disable Nginx buffering
    )


@app.post(
    "/ask-with-metadata",
    response_class=StreamingResponse,
    summary="Stream supervisor response with routing metadata",
    tags=["Streaming"],
)
async def ask_supervisor_with_metadata(
    payload: SupervisorRequest = Body(
        examples={
            "with_routing": {
                "summary": "Get answer with routing decision",
                "value": {
                    "question": "¿Qué especies están en peligro?",
                    "include_metadata": True,
                },
            }
        }
    ),
) -> StreamingResponse:
    """Stream supervisor response with routing decision metadata.

    This endpoint provides:
    - The routing decision (which agent was selected)
    - The reasoning behind the routing
    - The final answer from the selected agent

    Useful for debugging or displaying routing information in the UI.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    generator = _stream_supervisor_with_metadata(
        question=payload.question,
        include_metadata=payload.include_metadata,
        debug=payload.debug,
        recursion_limit=payload.recursion_limit,
    )
    return StreamingResponse(
        generator,
        media_type="application/x-ndjson; charset=utf-8",
        headers={"X-Accel-Buffering": "no"},
    )


@app.post(
    "/stream-sse",
    response_class=StreamingResponse,
    summary="Stream using Server-Sent Events format",
    tags=["Streaming"],
)
async def stream_sse(
    payload: SupervisorRequest = Body(
        examples={
            "sse_example": {
                "summary": "SSE format for browser EventSource",
                "value": {
                    "question": "¿Cuántas comunas tienen proyectos?",
                    "include_metadata": False,
                },
            }
        }
    ),
) -> StreamingResponse:
    """Stream supervisor response in Server-Sent Events (SSE) format.

    This format is compatible with browser EventSource API and provides
    better real-time streaming support for web applications.

    Each message is prefixed with "data: " and followed by double newline.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    generator = _stream_sse_format(
        question=payload.question,
        include_metadata=payload.include_metadata,
        debug=payload.debug,
        recursion_limit=payload.recursion_limit,
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post(
    "/ask-json",
    summary="Get complete response as JSON (non-streaming)",
    tags=["Non-Streaming"],
)
async def ask_json(payload: SupervisorRequest):
    """Get the complete supervisor response as a JSON object.

    This is a non-streaming endpoint that waits for the complete response
    before returning. Useful for clients that don't support streaming.
    """
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Collect the complete answer
        final_answer = None
        routing_info = {}

        if payload.include_metadata:
            async for update in stream_supervisor_with_metadata(
                question=payload.question,
                include_routing=True,
                debug=payload.debug,
                recursion_limit=payload.recursion_limit,
            ):
                if update["type"] == "routing":
                    routing_info = update["metadata"]
                elif update["type"] == "answer":
                    final_answer = update["content"]
        else:
            async for answer in stream_supervisor_updates(
                question=payload.question,
                debug=payload.debug,
                recursion_limit=payload.recursion_limit,
            ):
                final_answer = answer

        response = {
            "question": payload.question,
            "answer": final_answer,
            "status": "success" if final_answer else "no_answer",
        }

        if payload.include_metadata and routing_info:
            response["routing"] = routing_info

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {e!s}",
        )


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "src.API.supervisor_streaming_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
