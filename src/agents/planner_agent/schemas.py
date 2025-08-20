"""Schemas for the planner supervisor."""

# %%
from pydantic import BaseModel, Field


class Step(BaseModel):
    """A step in the plan."""

    instruction: str = Field(description="The instruction to be executed in this step")
    suggested_tool: str = Field(description="The tool suggested to execute this step")
    reasoning: str = Field(description="The reasoning for this step")
    result: str = Field(description="The result of the step")
    is_complete: bool = Field(description="Whether this step has been completed or not")


class Plan(BaseModel):
    """A plan for the agent."""

    goal: str = Field(description="The goal of the plan based on the user's request")
    steps: list[Step] = Field(
        description="List of steps to be executed to achieve the goal"
    )
    direct_response_to_the_user: str = Field(
        description="The direct response to a user trivial question. When no tools are needed."
    )

    @property
    def is_complete(self) -> bool:
        """Check if all steps are complete."""
        return all(step.is_complete for step in self.steps)
