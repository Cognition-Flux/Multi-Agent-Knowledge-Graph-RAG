# %%
from __future__ import annotations

import json

from langchain_core.tools import tool


try:  # Lazy-friendly import: avoid hard failure if Neo4j isn't configured
    from KnowledgeGraphDB.Neo4j_KG_creation.cypher_runner import run_cypher
except Exception:  # pragma: no cover - exercised via tests with monkeypatch

    def run_cypher(*_args, **_kwargs):  # type: ignore[no-redef]
        """Fallback stub when Neo4j isn't configured for tests.

        Raises at runtime if called to surface configuration issues quickly.
        """
        raise RuntimeError("Neo4j connection is not configured")


@tool(parse_docstring=True)
def list_proyectos_por_comuna_por_region(region: str) -> str:
    """List all projects with their communes for a given region.

    Args:
        region: Exact region name to match.

    Returns:
        str: JSON string with keys:
            - "region": la región de entrada
            - "proyectos": lista de objetos con campos "proyecto" y "comuna",
              ordenados por comuna y luego por proyecto. Ej.:
              [{"proyecto": "P1", "comuna": "C1"}, ...]
            - "count": número de elementos
            Si ocurre un error, retorna {"error": "..."}.
    """
    if not isinstance(region, str) or not region.strip():
        return json.dumps(
            {"error": "'region' must be a non-empty string"}, ensure_ascii=False
        )
    cypher = """
        MATCH (p:Project)-[:IN_REGION]->(r:Region {name: $region})
        MATCH (p)-[:IN_COMMUNE]->(c:Commune)
        RETURN p.name AS project_name, c.name AS commune_name
        """
    try:
        rows = run_cypher(cypher, {"region": region})
        # Collect (project, commune) pairs
        pairs: set[tuple[str, str]] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            project = row.get("project_name")
            commune = row.get("commune_name")
            if isinstance(project, str) and isinstance(commune, str):
                pairs.add((project, commune))

        # Sort by commune then project
        sorted_pairs = sorted(pairs, key=lambda pc: (pc[1], pc[0]))
        items = [{"proyecto": p, "comuna": c} for p, c in sorted_pairs]

        payload = {"region": region, "proyectos": items, "count": len(items)}
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return json.dumps({"error": f"{exc}"}, ensure_ascii=False)


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    REGION_COQUIMBO = "Región de Coquimbo"

    print(list_proyectos_por_comuna_por_region.invoke({"region": REGION_COQUIMBO}))
