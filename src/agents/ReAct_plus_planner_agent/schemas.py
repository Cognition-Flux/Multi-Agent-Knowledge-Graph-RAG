"""Schemas for the combined ReAct + Planner agent state."""

from typing import Annotated, Any

from langgraph.graph import MessagesState
from pydantic import Field

from src.agents.planner_agent.schemas import Plan
from src.agents.ReAct_agent.schemas import ToolResult


def update_plan_reducer(existing: Plan | None, new: Plan) -> Plan:
    """Reducer for updating the plan in the state."""
    if existing is None:
        return new
    return new


def append_results(existing: list[Any], new: list[Any]) -> list[Any]:
    """Reducer for appending results to the list."""
    return existing + new


class ReActPlusPlannerState(MessagesState):
    """Combined state for ReAct + Planner agent."""

    # Input
    question: str = Field(default="", description="The original question from the user")

    # Tools information
    available_tools: str = Field(
        default="", description="Description of available tools"
    )

    # Planning phase
    plan: Annotated[Plan | None, update_plan_reducer] = Field(
        default=None, description="The generated plan"
    )

    # Execution phase (from ReAct)
    current_step_index: int = Field(
        default=0, description="Current step being executed"
    )

    tool_results: Annotated[list[ToolResult], append_results] = Field(
        default_factory=list, description="Results from tool executions"
    )

    # Output
    final_answer: str = Field(default="", description="The final answer to return")

    # Status
    errors: Annotated[list[str], append_results] = Field(
        default_factory=list, description="Any errors encountered"
    )

    is_complete: bool = Field(
        default=False, description="Whether execution is complete"
    )
