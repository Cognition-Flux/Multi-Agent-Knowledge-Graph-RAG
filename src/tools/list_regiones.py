# %%
from KnowledgeGraphDB.Neo4j_KG_creation.cypher_runner import run_cypher


def list_regiones() -> str:
    """List all regions."""
    cypher = "MATCH (r:Region) RETURN r.name"
    return run_cypher(cypher)


if __name__ == "__main__":
    print(list_regiones())

# %%

# %%
