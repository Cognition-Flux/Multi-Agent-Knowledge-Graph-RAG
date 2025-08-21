"""Structured tool to list communes by region from the Neo4j knowledge graph.

Returns a JSON string with the following schema on success:

    {"region": "...", "comunas": ["...", "..."], "count": 2}

On invalid input or failure, returns a JSON error object:

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
def list_comunas_en_regiones(region: str) -> str:
    """List all unique commune names for a given region.

    Args:
        region: Exact region name to match.

    Returns:
        str: JSON string with keys:
            - "region": the input region string
            - "comunas": list of commune names (sorted ascending)
            - "count": number of communes
            If an error occurs or input is invalid, returns {"error": "..."}.
    """
    if not isinstance(region, str) or not region.strip():
        return json.dumps(
            {"error": "'region' must be a non-empty string"}, ensure_ascii=False
        )

    cypher = """
        MATCH (r:Region {name: $region})<-[:IN_REGION]-(p:Project)-[:IN_COMMUNE]->(c:Commune)
        RETURN DISTINCT c.name AS name
        ORDER BY name
        """
    try:
        rows = run_cypher(cypher, parameters={"region": region})
        comunas = [
            row.get("name") for row in rows if isinstance(row, dict) and row.get("name")
        ]
        comunas = sorted(comunas)
        payload = {"region": region, "comunas": comunas, "count": len(comunas)}
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"{exc}"}, ensure_ascii=False)


if __name__ == "__main__":  # pragma: no cover
    REGION_METROPOLITANA = "Región Metropolitana de Santiago"
    print(list_comunas_en_regiones.invoke({"region": REGION_METROPOLITANA}))
    REGION_COQUIMBO = "Región de Coquimbo"
    print(list_comunas_en_regiones.invoke({"region": REGION_COQUIMBO}))

# %%
