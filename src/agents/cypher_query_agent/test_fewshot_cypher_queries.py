# %%
import json
from datetime import date
from pathlib import Path

import yaml
from neo4j.exceptions import Neo4jError

from KnowledgeGraphDB.Neo4j_KG_creation.cypher_runner import close_driver, run_cypher


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code."""
    if isinstance(obj, date) or hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


# --- CONFIGURATION ---
# Select which queries to run from the 'fewshots.yaml' file.
# - To run all queries: slice(None)
# - To run the first 5 queries: slice(5)
# - To run queries from index 5 to 10: slice(5, 11)
# - To run only the query at index 4: 4
query_selection = slice(None)

yaml_file_path = Path(__file__).parent / "fewshots.yaml"

queries_to_run = []
if yaml_file_path.exists():
    with yaml_file_path.open(encoding="utf-8") as file:
        few_shots_data = yaml.safe_load(file)
        few_shots_list = few_shots_data.get("FEW_SHOTS_CYPHER_QUERY", [])
        if few_shots_list:
            for item in few_shots_list:
                if "pregunta" in item and "cypher_query" in item:
                    queries_to_run.append(
                        {
                            "pregunta": item["pregunta"],
                            "cypher_query": str(item["cypher_query"]),
                        }
                    )

# Apply the selection to determine which queries to execute
if isinstance(query_selection, int):
    if 0 <= query_selection < len(queries_to_run):
        selected_items = {query_selection: queries_to_run[query_selection]}
    else:
        print(
            f"Error: Index {query_selection} is out of bounds for "
            f"{len(queries_to_run)} queries."
        )
        selected_items = {}
elif isinstance(query_selection, slice):
    indices = range(*query_selection.indices(len(queries_to_run)))
    selected_items = {i: queries_to_run[i] for i in indices}
else:
    print("Error: Invalid selection type. Use an integer or a slice.")
    selected_items = {}

if selected_items:
    print(
        f"Found {len(queries_to_run)} total queries. "
        f"Running {len(selected_items)} selected queries..."
    )

    for index, data in selected_items.items():
        pregunta = data["pregunta"]
        cypher_query = data["cypher_query"]

        print(f"\n--- Query {index + 1}/{len(queries_to_run)} ---")
        print(f"Pregunta: {pregunta}")
        print("\nCypher Query:")
        print(cypher_query)

        try:
            result = run_cypher(cypher_query)
            print("\nResultado:")
            print(json.dumps(result, indent=2, ensure_ascii=False, default=json_serial))
        except Neo4jError as e:
            print(f"\nError al ejecutar la consulta: {e}")

        print("-" * 20)

    # Clean up the database connection
    close_driver()
    print("\nNeo4j driver closed.")
