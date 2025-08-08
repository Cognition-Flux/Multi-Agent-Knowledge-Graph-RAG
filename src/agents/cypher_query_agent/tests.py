"""Pytest for `create_dynamic_fewshooter`.

How to run:
- uv run -m pytest -q src/agents/cypher_query_agent/tests.py

This test patches embeddings to avoid external API/network calls.
It validates that examples are auto-detected from a YAML file, that
semantic selection returns k examples, and that prompt rendering works.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from src.agents.cypher_query_agent import (
    fewshooter_builder as fb,
    llm_chains,
    schemas,
)


class FakeEmbeddings:
    """Deterministic, in-memory embeddings for tests."""

    def __init__(self, *args, **kwargs) -> None:  # accept any ctor args
        pass

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 7), 0.0, 1.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text) % 7), 0.0, 1.0]


def test_create_dynamic_fewshooter_selects_examples(tmp_path, monkeypatch) -> None:
    # Arrange: YAML with Spanish keys (auto-detected by the builder)
    yaml_text = (
        "- pregunta: Cuantos proyectos hay?\n"
        "  cypher_query: |\n"
        "    MATCH (p:Project)\n"
        "    RETURN count(DISTINCT p) AS total\n"
        "- pregunta: proyectos en Antofagasta\n"
        "  cypher_query: |\n"
        "    MATCH (p:Project)-[:IN_REGION]->(:Region {name:'Región de Antofagasta'})\n"
        "    RETURN DISTINCT p.name AS project\n"
    )
    yaml_path = tmp_path / "fewshots_test.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    # Patch embeddings to use in-memory fake (no credentials/network)
    monkeypatch.setattr(fb, "AzureOpenAIEmbeddings", FakeEmbeddings, raising=True)
    monkeypatch.setattr(fb, "OpenAIEmbeddings", FakeEmbeddings, raising=True)

    # Act
    prompt = fb.create_dynamic_fewshooter(yaml_path=yaml_path, k=2)

    # Assert: correct type and selector exists
    assert isinstance(prompt, FewShotChatMessagePromptTemplate)
    selector = getattr(prompt, "example_selector", None)
    assert selector is not None

    # Assert: selecting examples returns exactly k and with required fields
    selected = selector.select_examples({"input": "Antofagasta"})
    assert len(selected) == 2
    for ex in selected:
        assert "input" in ex and str(ex["input"]).strip()
        assert "output" in ex and str(ex["output"]).strip()

    # Assert: rendering messages works end-to-end
    messages = prompt.format_messages(input="Antofagasta")
    assert len(messages) > 0


def test_cypher_query_chain_returns_valid_schema(monkeypatch) -> None:
    # Patch few-shots to avoid embeddings and YAML
    def _dummy_fs(k: int = 3, group: str | None = None):
        return ChatPromptTemplate.from_messages([("system", "[FEW-SHOTS HERE]")])

    class FakeLLM:
        def bind(self, **kwargs):
            return self

        def with_structured_output(self, schema):
            def _runner(_messages):
                # Return a valid CypherQuery regardless of prompt
                if schema is llm_chains.CypherQuery:
                    return llm_chains.CypherQuery(cypher_query="MATCH (n) RETURN n")
                raise AssertionError("Unexpected schema")

            return _runner

    monkeypatch.setattr(
        llm_chains, "create_dynamic_fewshooter", _dummy_fs, raising=True
    )
    monkeypatch.setattr(llm_chains, "get_llm", lambda: FakeLLM(), raising=True)

    chain = llm_chains.get_cypher_query_chain(k=2)
    out = chain.invoke({"input": "dummy"})

    assert isinstance(out, llm_chains.CypherQuery)
    assert out.cypher_query and "return" in out.cypher_query.lower()


def test_question_generation_chain_dedupes_and_parses(monkeypatch) -> None:
    # Patch few-shots to avoid embeddings and YAML
    def _dummy_fs(k: int = 3, group: str | None = None):
        return ChatPromptTemplate.from_messages([("system", "[FEW-SHOTS HERE]")])

    class FakeLLM:
        def bind(self, **kwargs):
            return self

        def with_structured_output(self, schema):
            def _runner(_messages):
                if schema is llm_chains.GeneratedQueries:
                    # Provide duplicates to test model validator dedupe
                    return schemas.GeneratedQueries(
                        queries_list=[
                            schemas.OneQuery(query_str="Q1"),
                            schemas.OneQuery(query_str="q1"),
                        ]
                    )
                raise AssertionError("Unexpected schema")

            return _runner

    monkeypatch.setattr(
        llm_chains, "create_dynamic_fewshooter", _dummy_fs, raising=True
    )
    monkeypatch.setattr(llm_chains, "get_llm", lambda: FakeLLM(), raising=True)

    chain = llm_chains.get_question_generation_chain(k=2)
    out = chain.invoke({"input": "dummy"})

    assert isinstance(out, llm_chains.GeneratedQueries)
    # Dedup should produce only one unique query
    assert len(out.queries_list) == 1
    assert out.queries_list[0].query_str.lower() == "q1"
