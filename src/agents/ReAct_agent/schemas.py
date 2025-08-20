"""Schemas for the ReAct agent state and tool responses."""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from src.agents.planner_agent.schemas import Plan


def update_plan_reducer(existing: Plan | None, new: Plan) -> Plan:
    """Reducer for updating the plan in the state."""
    if existing is None:
        return new
    # Merge updates from the new plan
    return new


def update_current_step_reducer(existing: int, new: int) -> int:
    """Reducer for updating the current step index."""
    return new


def append_results(
    existing: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Reducer for appending results to the list."""
    return existing + new


class ToolResult(BaseModel):
    """Result from a tool execution."""

    tool_name: str = Field(description="Name of the tool that was executed")
    step_index: int = Field(description="Index of the step that was executed")
    result: Any = Field(description="Result from the tool execution")
    error: str | None = Field(
        default=None, description="Error message if execution failed"
    )


class ReActState(MessagesState):
    """State for the ReAct agent."""

    # Input plan from the planner
    plan: Annotated[Plan, update_plan_reducer] = Field(
        default_factory=lambda: Plan(goal="", steps=[], direct_response_to_the_user="")
    )

    # Current step being executed
    current_step_index: Annotated[int, update_current_step_reducer] = Field(default=0)

    # Results from tool executions
    tool_results: Annotated[list[ToolResult], append_results] = Field(
        default_factory=list
    )

    # Final answer to return
    final_answer: str = Field(default="")

    # Error tracking
    errors: Annotated[list[str], append_results] = Field(default_factory=list)

    # Execution complete flag
    is_complete: bool = Field(default=False)
