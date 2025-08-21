"""Structured tool for listing regions from the Neo4j knowledge graph.

This tool executes a safe, parameterless Cypher query and returns a JSON string
with the following schema on success:

    {"regions": ["...", "..."], "count": 2}

On failure, it returns a JSON error object:

    {"error": "..."}
"""

# %%
from __future__ import annotations

import json

from langchain_core.tools import tool


try:  # Lazy-friendly import: avoid hard failure if Neo4j isn't configured
    from KnowledgeGraphDB.Neo4j_KG_creation.cypher_runner import run_cypher
except Exception:  # pragma: no cover - exercised via tests with monkeypatch

    def run_cypher(*_args, **_kwargs):  # type: ignore[no-redef]
        raise RuntimeError("Neo4j connection is not configured")


@tool(parse_docstring=True)
def list_regiones() -> str:
    """List all unique region names.

    Returns:
        str: JSON string with the keys:
            - "regions": list of region names (sorted ascending)
            - "count": number of regions
            If an error occurs, returns {"error": "..."}.
    """
    cypher = """
        MATCH (r:Region)
        RETURN DISTINCT r.name AS name
        ORDER BY name
        """
    try:
        rows = run_cypher(cypher)
        regions = [
            row.get("name") for row in rows if isinstance(row, dict) and row.get("name")
        ]
        payload = {"regions": regions, "count": len(regions)}
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"{exc}"}, ensure_ascii=False)


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    print(list_regiones.invoke({}))

# %%
