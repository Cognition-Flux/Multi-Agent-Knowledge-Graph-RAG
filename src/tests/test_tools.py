"""Unit tests for structured tools in src.tools.

These tests monkeypatch the underlying `run_cypher` helper to avoid requiring a
live Neo4j connection and to exercise success and error paths.
"""

from __future__ import annotations

import json


def test_list_regiones_success(monkeypatch) -> None:
    """list_regiones should return distinct, sorted region names and a count."""
    import src.tools.list_regiones as mod

    def fake_run_cypher(query: str, parameters=None, *, close_after: bool = False):
        assert "MATCH (r:Region)" in query
        return [{"name": "B"}, {"name": "A"}, {"name": "A"}]

    monkeypatch.setattr(mod, "run_cypher", fake_run_cypher)

    result_json = mod.list_regiones.invoke({})
    payload = json.loads(result_json)
    assert payload == {"regions": ["A", "B"], "count": 2}


def test_list_comunas_success(monkeypatch) -> None:
    """list_comunas should return distinct, sorted commune names and a count."""
    import src.tools.list_comunas as mod

    def fake_run_cypher(query: str, parameters=None, *, close_after: bool = False):
        assert "MATCH (c:Commune)" in query
        return [{"name": "X"}, {"name": "Y"}, {"name": "X"}]

    monkeypatch.setattr(mod, "run_cypher", fake_run_cypher)

    result_json = mod.list_comunas.invoke({})
    payload = json.loads(result_json)
    assert payload == {"comunas": ["X", "Y"], "count": 2}


def test_list_comunas_en_regiones_success(monkeypatch) -> None:
    """list_comunas_en_regiones should safely parametrize and return results."""
    import src.tools.list_comunas_en_regiones as mod

    calls: dict[str, object] = {}

    def fake_run_cypher(query: str, parameters=None, *, close_after: bool = False):
        calls["query"] = query
        calls["parameters"] = parameters
        assert "{name: $region}" in query
        assert isinstance(parameters, dict) and parameters.get("region") == "R1"
        return [{"name": "c2"}, {"name": "c1"}, {"name": "c1"}]

    monkeypatch.setattr(mod, "run_cypher", fake_run_cypher)

    result_json = mod.list_comunas_en_regiones.invoke({"region": "R1"})
    payload = json.loads(result_json)
    assert payload == {"region": "R1", "comunas": ["c1", "c2"], "count": 2}


def test_list_comunas_en_regiones_invalid(monkeypatch) -> None:
    """Invalid region should return an error JSON object."""
    import src.tools.list_comunas_en_regiones as mod

    result_json = mod.list_comunas_en_regiones.invoke({"region": "   "})
    payload = json.loads(result_json)
    assert "error" in payload


def test_get_tools_exports() -> None:
    """get_tools should expose the three structured tools."""
    from src.tools import (
        get_tools,
        list_comunas,
        list_comunas_en_regiones,
        list_regiones,
    )

    tools = get_tools()
    names = {t.name for t in tools}
    assert (
        list_regiones in tools
        and list_comunas in tools
        and list_comunas_en_regiones in tools
    )
    assert names == {"list_regiones", "list_comunas", "list_comunas_en_regiones"}
