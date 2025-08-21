# %%
from KnowledgeGraphDB.Neo4j_KG_creation.cypher_runner import run_cypher


def list_comunas() -> str:
    """List all comunas in a given region."""
    cypher = """
    /* all comunes names */

    MATCH (c:Commune)
    RETURN c.name
    """
    return run_cypher(cypher)


if __name__ == "__main__":
    comunas = list_comunas()
    print(comunas)

# %%

# %%
