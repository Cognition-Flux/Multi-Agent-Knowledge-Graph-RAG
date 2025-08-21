# %%
from KnowledgeGraphDB.Neo4j_KG_creation.cypher_runner import run_cypher


def list_comunas_en_regiones(region: str) -> str:
    """List all comunas in a given region."""
    cypher = f"""
    /* all the comunes in region: {region} */

    MATCH (r:Region {{name: '{region}'}})<-[:IN_REGION]-(p:Project)-[:IN_COMMUNE]->(c:Commune)
    RETURN c.name AS commune
    """
    return run_cypher(cypher)


if __name__ == "__main__":
    REGION_METROPOLITANA = "Región Metropolitana de Santiago"
    comunas = list_comunas_en_regiones(REGION_METROPOLITANA)
    print(comunas)
    REGION_COQUIMBO = "Región de Coquimbo"
    comunas = list_comunas_en_regiones(REGION_COQUIMBO)
    print(comunas)

# %%
