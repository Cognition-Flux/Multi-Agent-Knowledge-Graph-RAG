"""Schemas for the reasoning agent."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class ReasoningTask(BaseModel):
    """A reasoning task to be performed."""

    task_type: Literal[
        "summarize",
        "describe",
        "reflect",
        "analyze",
        "think",
        "read",
        "judge",
        "interpret",
        "synthesize",
        "compare",
        "evaluate",
    ] = Field(description="The type of reasoning task to perform")

    focus: str = Field(description="What to focus on during the reasoning task")

    context: str = Field(
        default="", description="Additional context for the reasoning task"
    )


class ReasoningResponse(BaseModel):
    """Response from a reasoning task."""

    reasoning: str = Field(description="The reasoning process and thoughts")

    conclusion: str = Field(description="The final conclusion or output")

    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence level in the conclusion (0-1)",
    )

    key_points: list[str] = Field(
        default_factory=list, description="Key points identified during reasoning"
    )


class ReasoningState(MessagesState):
    """State for the reasoning agent."""

    # Input
    instruction: str = Field(
        default="", description="The reasoning instruction to execute"
    )

    current_results: list[Any] = Field(
        default_factory=list, description="Current results from previous steps"
    )

    partial_results: list[Any] = Field(
        default_factory=list, description="Partial results to consider"
    )

    # Processing
    reasoning_task: ReasoningTask | None = Field(
        default=None, description="The parsed reasoning task"
    )

    # Output
    reasoning_response: ReasoningResponse | None = Field(
        default=None, description="The reasoning response"
    )

    final_output: str = Field(default="", description="The final output text")
