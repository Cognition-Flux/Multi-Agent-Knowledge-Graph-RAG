"""Example API usage of the supervisor graph streamer.

This file demonstrates how to integrate the async streamer with FastAPI
for real-time streaming responses, similar to the implementation in
KnowledgeGraphDB/Neo4j_KG_creation/API_for_graph.py.
"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.graph_streamers.async_stream_by_updates import (
    stream_supervisor_updates,
    stream_supervisor_with_metadata,
)


# Initialize FastAPI app
app = FastAPI(
    title="Supervisor Agent API",
    description="API for streaming responses from the supervisor agent",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QuestionRequest(BaseModel):
    """Request model for questions."""

    question: str
    include_metadata: bool = False
    debug: bool = False
    recursion_limit: int = 50


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str


# --------------------------------------------------------------------------- #
# API Endpoints
# --------------------------------------------------------------------------- #


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="supervisor-agent-streamer",
    )


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Stream the supervisor agent's response to a question.

    The supervisor will route the question to the appropriate agent:
    - Cypher agent for metadata queries
    - GraphRAG agent for content queries

    Returns a streaming response with the final answer.
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    async def generate():
        """Generate streaming response."""
        try:
            async for answer in stream_supervisor_updates(
                question=request.question,
                debug=request.debug,
                recursion_limit=request.recursion_limit,
            ):
                # Wrap answer in SSE format for better client compatibility
                yield f"data: {json.dumps({'answer': answer})}\n\n"
        except Exception as e:
            error_msg = f"Error processing question: {e!s}"
            yield f"data: {json.dumps({'error': error_msg})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )


@app.post("/ask-with-metadata")
async def ask_question_with_metadata(request: QuestionRequest):
    """Stream the supervisor agent's response with routing metadata.

    This endpoint provides additional information about:
    - Which agent was selected (cypher or graphrag)
    - The reasoning behind the routing decision
    - The final answer

    Useful for debugging or showing routing details in the UI.
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    async def generate():
        """Generate streaming response with metadata."""
        try:
            async for update in stream_supervisor_with_metadata(
                question=request.question,
                include_routing=request.include_metadata,
                debug=request.debug,
                recursion_limit=request.recursion_limit,
            ):
                # Stream each update as a JSON line
                yield f"data: {json.dumps(update)}\n\n"
        except Exception as e:
            error_update = {
                "type": "error",
                "content": str(e),
                "metadata": {},
            }
            yield f"data: {json.dumps(error_update)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/ask-json")
async def ask_question_json(request: QuestionRequest):
    """Non-streaming endpoint that returns the complete answer as JSON.

    This is useful for clients that don't support streaming or when
    you want the complete response at once.
    """
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # Collect the complete answer
        final_answer = None
        async for answer in stream_supervisor_updates(
            question=request.question,
            debug=request.debug,
            recursion_limit=request.recursion_limit,
        ):
            final_answer = answer

        if final_answer:
            return {
                "question": request.question,
                "answer": final_answer,
                "status": "success",
            }
        else:
            return {
                "question": request.question,
                "answer": None,
                "status": "no_answer",
                "message": "No answer was generated",
            }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {e!s}",
        )


# --------------------------------------------------------------------------- #
# Run the API (for development)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import uvicorn

    # Run with: python src/graph_streamers/example_api_usage.py
    # Or: uvicorn src.graph_streamers.example_api_usage:app --reload
    uvicorn.run(
        "src.graph_streamers.example_api_usage:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
