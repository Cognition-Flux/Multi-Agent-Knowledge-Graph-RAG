"""Este script ejecuta las consultas de ejemplo de sample_queries.yaml y muestra los resultados."""
# %%

from pathlib import Path

import yaml

from KnowledgeGraphDB.Neo4j_KG_creation.cypher_runner import run_cypher


yaml_path = Path(__file__).with_name("sample_queries_cypher_agent.yaml")


sample_queries = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

# cada elemento: {'pregunta': str, 'cypher_query': str}

# --------------------------------------------------------------------------- #
# Ejecutar cada consulta y mostrar resultados
# --------------------------------------------------------------------------- #

for item in sample_queries:
    title = item["pregunta"]
    cypher = item["cypher_query"]
    print(f"\n📌 {title}\nCypher: {cypher.strip()}\n→ Resultados:")
    rows = run_cypher(cypher)
    for row in rows:
        print(row)
