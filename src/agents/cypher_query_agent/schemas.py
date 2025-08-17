"""Pydantic schemas for the Cypher Query agent.

- Centralizes data models to be imported by chains and tests.
"""

from __future__ import annotations

from typing import Annotated

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, field_validator, model_validator

from src.agents.cypher_query_agent.reducers import reduce_lists


class OneQuery(BaseModel):
    """One query (single textual query)."""

    query_str: str = Field(min_length=1, alias="query")

    model_config = {
        "extra": "forbid",
        "populate_by_name": True,
    }

    @field_validator("query_str")
    @classmethod
    def _clean_and_validate_query(cls, value: str) -> str:
        cleaned = str(value).strip()
        # Remove generic triple-backtick fences if present
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("```", "").strip()
        if not cleaned:
            raise ValueError("query_str must be non-empty")
        return cleaned


class GeneratedQueries(BaseModel):
    """A list of generated queries."""

    queries_list: list[OneQuery] = Field(default_factory=list, alias="queries")

    model_config = {
        "extra": "forbid",
        "populate_by_name": True,
    }

    @model_validator(mode="after")
    def _dedupe_and_validate(self) -> GeneratedQueries:
        # Ensure non-empty
        if not self.queries_list:
            raise ValueError("queries_list must contain at least one query")
        # Dedupe preserving order by normalized query_str
        seen: set[str] = set()
        unique: list[OneQuery] = []
        for item in self.queries_list:
            key = item.query_str.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(item)
        if not unique:
            raise ValueError("queries_list normalization produced empty set")
        self.queries_list = unique
        return self


class CypherQuery(BaseModel):
    """Cypher query agent output schema."""

    cypher_query: str = Field(min_length=1)

    model_config = {
        "extra": "forbid",
    }

    @field_validator("cypher_query")
    @classmethod
    def _strip_and_validate(cls, value: str) -> str:
        """Sanitise the query in case the LLM returned it inside markdown fences."""
        cleaned = str(value).strip()
        if cleaned.startswith("```"):
            # Remove leading/trailing code fences
            stripped = cleaned.strip("`").strip()
            # If language identifier present (e.g. ```cypher), drop first line
            if "\n" in stripped:
                first_line, rest = stripped.split("\n", 1)
                cleaned = rest if first_line.lower().startswith("cypher") else stripped
            else:
                cleaned = stripped
        if not cleaned:
            raise ValueError("cypher_query must be non-empty")
        return cleaned


class Neo4jQueryState(MessagesState):
    """State of the Neo4j Graph RAG."""

    question: str = Field(default_factory=lambda: "")
    generated_questions: GeneratedQueries = Field(
        default_factory=lambda: GeneratedQueries(queries_list=[])
    )
    query: str = Field(default_factory=lambda: "")
    cypher_query: str = Field(default_factory=lambda: "")
    cypher_queries: Annotated[list[str], reduce_lists] = Field(default_factory=list)
    results: Annotated[list[str], reduce_lists] = Field(default_factory=list)


class Answer(BaseModel):
    """Answer schema."""

    answer: str = Field(description="The answer to the question.")


__all__ = [
    "Answer",
    "CypherQuery",
    "GeneratedQueries",
    "Neo4jQueryState",
    "OneQuery",
]
