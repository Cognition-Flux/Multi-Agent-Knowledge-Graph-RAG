"""This module contains the schemas for the user question augmentation agent."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


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


__all__ = [
    "GeneratedQueries",
    "OneQuery",
]
