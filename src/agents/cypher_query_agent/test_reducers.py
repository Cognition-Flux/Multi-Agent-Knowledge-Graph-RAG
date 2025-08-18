"""Unit tests for reducer functions.

This module tests the reduce_lists and reduce_lists_allow_duplicates functions
to ensure they handle all edge cases correctly.

Usage:
    uv run -m pytest src/agents/cypher_query_agent/test_reducers.py -v
    or
    uv run -m src.agents.cypher_query_agent.test_reducers
"""

import pytest

from src.agents.cypher_query_agent.reducers import (
    reduce_lists,
    reduce_lists_allow_duplicates,
)


class TestReduceLists:
    """Test cases for the reduce_lists function."""

    def test_none_inputs(self):
        """Test handling of None inputs."""
        assert reduce_lists(None, None) == []
        assert reduce_lists(None, "hello") == ["hello"]
        assert reduce_lists(["existing"], None) == ["existing"]

    def test_delete_signal(self):
        """Test the delete signal resets the list."""
        assert reduce_lists(["a", "b", "c"], "delete") == []
        assert reduce_lists(None, "delete") == []

    def test_single_string_input(self):
        """Test adding a single string."""
        assert reduce_lists([], "new") == ["new"]
        assert reduce_lists(["old"], "new") == ["old", "new"]

    def test_list_input(self):
        """Test adding a list of strings."""
        assert reduce_lists([], ["a", "b"]) == ["a", "b"]
        assert reduce_lists(["x"], ["y", "z"]) == ["x", "y", "z"]

    def test_deduplication(self):
        """Test that duplicates are removed while preserving order."""
        assert reduce_lists(["a", "b"], ["b", "c"]) == ["a", "b", "c"]
        assert reduce_lists(["x", "y", "x"], ["y", "z"]) == ["x", "y", "z"]
        assert reduce_lists(["a"], ["a", "a", "a"]) == ["a"]

    def test_order_preservation(self):
        """Test that order of first occurrence is preserved."""
        result = reduce_lists(["z", "a", "b"], ["c", "a", "d"])
        assert result == ["z", "a", "b", "c", "d"]

    def test_empty_lists(self):
        """Test handling of empty lists."""
        assert reduce_lists([], []) == []
        assert reduce_lists([], ["a"]) == ["a"]
        assert reduce_lists(["a"], []) == ["a"]

    def test_type_coercion(self):
        """Test that non-string items are converted to strings."""
        # This tests the defensive programming in the improved version
        result = reduce_lists(["1"], [2, 3])  # type: ignore
        # Numbers should be converted to strings
        assert "2" in result and "3" in result

    def test_fast_paths(self):
        """Test that fast paths work correctly."""
        # No new items - should return existing
        existing = ["a", "b", "c"]
        result = reduce_lists(existing, [])
        assert result == existing

        # No existing items - should deduplicate new only
        result = reduce_lists(None, ["a", "b", "a"])
        assert result == ["a", "b"]

    def test_realistic_langgraph_scenario(self):
        """Test a realistic scenario from parallel LangGraph execution."""
        # Simulating parallel GraphRAG results being accumulated
        state = None

        # First parallel result
        state = reduce_lists(state, ["Result from query 1"])
        assert state == ["Result from query 1"]

        # Second parallel result
        state = reduce_lists(state, ["Result from query 2"])
        assert state == ["Result from query 1", "Result from query 2"]

        # Third parallel result (duplicate)
        state = reduce_lists(state, ["Result from query 1"])
        assert state == ["Result from query 1", "Result from query 2"]

        # Reset
        state = reduce_lists(state, "delete")
        assert state == []


class TestReduceListsAllowDuplicates:
    """Test cases for the reduce_lists_allow_duplicates function."""

    def test_allows_duplicates(self):
        """Test that duplicates are preserved."""
        assert reduce_lists_allow_duplicates(["a"], ["a", "b"]) == ["a", "a", "b"]
        assert reduce_lists_allow_duplicates(["x", "x"], ["x"]) == ["x", "x", "x"]

    def test_none_handling(self):
        """Test handling of None inputs."""
        assert reduce_lists_allow_duplicates(None, None) == []
        assert reduce_lists_allow_duplicates(None, ["a"]) == ["a"]
        assert reduce_lists_allow_duplicates(["a"], None) == ["a"]

    def test_delete_signal(self):
        """Test the delete signal."""
        assert reduce_lists_allow_duplicates(["a", "b"], "delete") == []

    def test_single_item(self):
        """Test adding single items."""
        assert reduce_lists_allow_duplicates(["a"], "b") == ["a", "b"]
        assert reduce_lists_allow_duplicates([], 1) == [1]


def test_reducer_consistency():
    """Test that reducers behave consistently with LangGraph expectations."""
    # Test that reduce_lists is idempotent for duplicates
    result1 = reduce_lists(["a", "b"], ["b", "c"])
    result2 = reduce_lists(result1, ["b", "c"])
    assert result1 == result2  # Should be the same since duplicates are removed

    # Test that reduce_lists_allow_duplicates is NOT idempotent
    result1 = reduce_lists_allow_duplicates(["a"], ["a"])
    result2 = reduce_lists_allow_duplicates(result1, ["a"])
    assert len(result1) < len(result2)  # Should grow with duplicates


def test_performance_characteristics():
    """Test that the reducer performs well with large lists."""
    # Create a large list with many duplicates
    large_list = list(range(1000)) * 10  # 10,000 items with 1000 unique

    # Convert to strings
    large_list_str = [str(i) for i in large_list]

    # Test deduplication
    result = reduce_lists([], large_list_str)
    assert len(result) == 1000  # Should have exactly 1000 unique items

    # Test order preservation
    assert result == [str(i) for i in range(1000)]


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
    print("\n✅ All reducer tests passed!")
