"""Run tests.

- uv run -m pytest -q src/agents/cypher_query_agent/tests.py
- uv run src/agents/cypher_query_agent/fewshooter_builder.py
"""

# %%
from collections.abc import Iterable
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings


load_dotenv(override=True)

# Default path to the few-shots YAML colocated with this module
DEFAULT_FEWSHOTS_PATH = Path(__file__).parent / "fewshots.yaml"


def create_dynamic_fewshooter(
    yaml_path: Path | None = None,
    input_key: str = "input",
    output_key: str = "output",
    *,
    k: int = 2,
    selector_input_variable: str = "input",
) -> FewShotChatMessagePromptTemplate:
    """Create a dynamic few-shot chat prompt template using semantic selection.

    - Loads examples from a YAML list of dicts.
    - Auto-detects input/output keys if defaults are not present.
    - Skips incomplete items safely.

    Args:
        yaml_path: Optional path to YAML. Defaults to `fewshots.yaml` in this module.
        input_key: Preferred input field name if present.
        output_key: Preferred output field name if present.
        k: Number of examples to select.
        selector_input_variable: The variable name whose value will drive example selection.

    Returns:
        FewShotChatMessagePromptTemplate ready to be composed in a chat prompt.
    """
    yaml_path = yaml_path or DEFAULT_FEWSHOTS_PATH

    # Read and validate YAML structure
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"YAML must be a list of objects: {yaml_path}")
    rows: list[dict] = [row for row in data if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"No dictionary examples found in YAML file: {yaml_path}")

    # Auto-detect best matching key pair by frequency
    candidate_pairs: list[tuple[str, str]] = [
        (input_key, output_key),
        ("pregunta", "cypher_query"),  # Spanish dataset
        ("question", "answer"),  # Common fallback
    ]

    def count_pair(items: Iterable[dict], inp: str, out: str) -> int:
        return sum(1 for it in items if inp in it and out in it)

    pair_counts = [(pair, count_pair(rows, *pair)) for pair in candidate_pairs]
    pair_counts.sort(key=lambda x: x[1], reverse=True)
    best_pair, best_count = pair_counts[0]
    if best_count == 0:
        available_keys = sorted({k for it in rows for k in it.keys()})
        raise KeyError(
            f"Could not infer input/output keys for {yaml_path}. "
            f"Tried {candidate_pairs}. Available keys: {available_keys}"
        )
    source_input_key, source_output_key = best_pair

    # Build normalized examples; skip incomplete ones
    examples = [
        {
            "input": str(it[source_input_key]).strip(),
            "output": str(it[source_output_key]).strip(),
        }
        for it in rows
        if source_input_key in it
        and source_output_key in it
        and str(it[source_input_key]).strip()
        and str(it[source_output_key]).strip()
    ]
    if not examples:
        raise ValueError(f"No valid examples after normalization from: {yaml_path}")

    # Prepare texts to vectorize (explicit order for stability)
    to_vectorize = [f"{ex['input']}\n{ex['output']}" for ex in examples]

    # Initialize embeddings with robust fallback
    embeddings = None
    init_errors: list[str] = []
    try:
        embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large")
    except Exception as exc:
        init_errors.append(f"AzureOpenAIEmbeddings failed: {exc}")
    if embeddings is None:
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        except Exception as exc:
            init_errors.append(f"OpenAIEmbeddings failed: {exc}")
            raise RuntimeError(
                "No embeddings backend available. Set Azure OpenAI or OpenAI credentials. "
                + "; ".join(init_errors)
            ) from exc

    # Build selector with safe k
    effective_k = max(1, min(k, len(examples)))
    vectorstore = InMemoryVectorStore.from_texts(
        to_vectorize, embeddings, metadatas=examples
    )
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=effective_k,
    )

    return FewShotChatMessagePromptTemplate(
        # The input variables select the values to pass to the example_selector
        input_variables=[selector_input_variable],
        example_selector=example_selector,
        # Define how each example will be formatted.
        # In this case, each example will become 2 messages: 1 human, and 1 ai
        example_prompt=ChatPromptTemplate.from_messages(
            [("human", "{input}"), ("ai", "{output}")]
        ),
    )


if __name__ == "__main__":
    SYSTEM_PROMPT = (
        "You are a helpful assistant that can answer questions about the graph."
    )
    FEW_SHOT_PROMPT = create_dynamic_fewshooter(k=1)
    TEST_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "system",
                "## A continuación, ejemplos de preguntas y respuestas parecidas:",
            ),
            FEW_SHOT_PROMPT,
            ("system", "## A continuación, el requerimiento del usuario:"),
            ("human", "{input}"),
        ]
    )
    # Demo: inspect which few-shot examples were selected for a given input
    DEMO_INPUT = "proyectos en las comunas Antofagasta o Mejillones"
    SELECTOR = getattr(FEW_SHOT_PROMPT, "example_selector", None)

    def _indent(text: str, prefix: str = "    ") -> str:
        return "\n".join(prefix + line for line in str(text).splitlines())

    def _section(title: str) -> None:
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

    if SELECTOR is not None:
        selected = SELECTOR.select_examples({"input": DEMO_INPUT})

        _section("Entrada de demostración")
        print(DEMO_INPUT)

        _section(f"Ejemplos few-shot seleccionados (k={len(selected)})")
        for idx, ex in enumerate(selected, start=1):
            INPUT_TEXT = str(ex.get("input", "")).strip()
            OUTPUT_TEXT = str(ex.get("output", "")).strip()
            print(f"[{idx}] INPUT:")
            print(_indent(INPUT_TEXT))
            print("    OUTPUT:")
            print(_indent(OUTPUT_TEXT))
            print("-" * 80)

        _section("Prompt final (mensajes en orden)")
        messages = TEST_PROMPT.format_messages(input=DEMO_INPUT)
        for i, msg in enumerate(messages, start=1):
            content = getattr(msg, "content", "")
            print(f"{i:02d}. [{msg.type}]")
            print(_indent(content))
            print()
    else:
        print("No example_selector found on FEW_SHOT_PROMPT.")

# %%
