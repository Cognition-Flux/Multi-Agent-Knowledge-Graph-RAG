"""Reducer functions for LangGraph state management.

This module provides reducer functions used by LangGraph to manage state updates,
particularly for handling parallel execution results and list accumulation.
"""

from __future__ import annotations

import logging
from typing import Any, Literal


# Configure logging
logging.basicConfig(level=logging.WARNING)


def reduce_lists(
    existing: list[str] | None,
    new: list[str] | str | Literal["delete"] | None,
) -> list[str]:
    """Combine two lists of strings in a robust way.

    This reducer is designed for LangGraph state management, particularly for
    accumulating results from parallel node executions while maintaining uniqueness
    and preserving order.

    Args:
        existing: The current list of strings in the state (can be None).
        new: New item(s) to add - can be:
            - A single string
            - A list of strings
            - The literal "delete" to reset the list
            - None (treated as empty list)

    Returns:
        A deduplicated list containing all unique items from both existing and new,
        preserving the order of first occurrence.

    Behavior:
        • If *new* is the literal ``"delete"`` → returns an empty list (reset).
        • If either argument is ``None`` → treats it as an empty list.
        • Accepts *new* as a single string or list of strings.
        • Ensures the returned list has **unique items preserving order**.
        • Handles type errors gracefully with logging.

    Examples:
        >>> reduce_lists(None, "hello")
        ['hello']
        >>> reduce_lists(["a", "b"], ["b", "c"])
        ['a', 'b', 'c']
        >>> reduce_lists(["x", "y"], "delete")
        []
    """
    # Reset signal - special case for clearing the list
    if new == "delete":
        return []

    # Normalize existing input
    if existing is None:
        existing = []
    elif not isinstance(existing, list):
        # Defensive: ensure existing is a list
        try:
            existing = list(existing)
        except (TypeError, ValueError) as exc:
            logging.warning(
                "reduce_lists: Invalid existing type %s, treating as empty: %s",
                type(existing).__name__,
                exc,
            )
            existing = []

    # Normalize new input
    new_items: list[str] = []
    if new is None:
        new_items = []
    elif isinstance(new, str):
        new_items = [new]
    elif isinstance(new, list):
        new_items = new
    else:
        # Try to convert to list if it's iterable
        try:
            new_items = list(new)
        except (TypeError, ValueError) as exc:
            logging.warning(
                "reduce_lists: Invalid new type %s, treating as empty: %s",
                type(new).__name__,
                exc,
            )
            new_items = []

    # Fast path: if no new items, return existing as-is
    if not new_items:
        return existing

    # Fast path: if no existing items, deduplicate only new items
    if not existing:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in new_items:
            # Ensure items are strings
            str_item = str(item) if not isinstance(item, str) else item
            if str_item not in seen:
                seen.add(str_item)
                deduped.append(str_item)
        return deduped

    # Combine and deduplicate while preserving order
    combined = existing + new_items
    seen = set()
    deduped = []

    for item in combined:
        # Ensure items are strings
        str_item = str(item) if not isinstance(item, str) else item
        if str_item not in seen:
            seen.add(str_item)
            deduped.append(str_item)

    return deduped


def reduce_lists_allow_duplicates(
    existing: list[Any] | None,
    new: list[Any] | Any | Literal["delete"] | None,
) -> list[Any]:
    """Combine two lists allowing duplicates (simple concatenation).

    This is an alternative reducer for cases where duplicate values should be preserved.
    Useful for aggregating all raw results before post-processing.

    Args:
        existing: The current list in the state (can be None).
        new: New item(s) to add.

    Returns:
        A list containing all items from both existing and new.

    Examples:
        >>> reduce_lists_allow_duplicates(["a"], ["a", "b"])
        ['a', 'a', 'b']
    """
    if new == "delete":
        return []

    if existing is None:
        existing = []

    if new is None:
        return existing
    elif not isinstance(new, list):
        return existing + [new]
    else:
        return existing + new


__all__ = ["reduce_lists", "reduce_lists_allow_duplicates"]
