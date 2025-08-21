"""Structured tool to list all communes from the Neo4j knowledge graph.

Returns a JSON string with the following schema on success:

    {"comunas": ["...", "..."], "count": 2}

On failure, returns a JSON error object:

    {"error": "..."}
"""

# %%
from __future__ import annotations

import json

from langchain_core.tools import tool


try:  # Lazy-friendly import so tests can monkeypatch without Neo4j available
    from KnowledgeGraphDB.Neo4j_KG_creation.cypher_runner import run_cypher
except Exception:  # pragma: no cover

    def run_cypher(*_args, **_kwargs):  # type: ignore[no-redef]
        raise RuntimeError("Neo4j connection is not configured")


@tool(parse_docstring=True)
def list_comunas() -> str:
    """List all unique commune names.

    Returns:
        str: JSON string with keys:
            - "comunas": list of commune names (sorted ascending)
            - "count": number of communes
            If an error occurs, returns {"error": "..."}.
    """
    cypher = """
        MATCH (c:Commune)
        RETURN DISTINCT c.name AS name
        ORDER BY name
        """
    try:
        rows = run_cypher(cypher)
        comunas = [
            row.get("name") for row in rows if isinstance(row, dict) and row.get("name")
        ]
        payload = {"comunas": comunas, "count": len(comunas)}
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"{exc}"}, ensure_ascii=False)


if __name__ == "__main__":  # pragma: no cover
    print(list_comunas.invoke({}))

# %%
