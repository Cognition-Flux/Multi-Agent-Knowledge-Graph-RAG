# %%
from __future__ import annotations

import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.cohere import CohereEmbeddings
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index
from neo4j_graphrag.retrievers import HybridCypherRetriever


# --------------------------------------------------------------------------- #
# 1) Entorno e índices
# --------------------------------------------------------------------------- #

load_dotenv(override=True)

NEO4J_USERNAME = os.getenv("NEO4J_USERNAME_UPGRADED")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD_UPGRADED")
NEO4J_URI = os.getenv("NEO4J_CONNECTION_URI_UPGRADED")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
# Verificamos conectividad sin cerrar el driver prematuramente.
driver.verify_connectivity()

# Embeddings Cohere (mismo modelo que en la construcción del KG)
embedder = CohereEmbeddings(model="embed-v4.0", api_key=os.getenv("COHERE_API_KEY"))

# Nombre de índices usados para nodos :Chunk creados por SimpleKGPipeline
vector_index_name = "chunkEmbedding"
fulltext_index_name = "chunkFulltext"

# Dimensionalidad inferida dinámicamente (solo una vez)
try:
    VECTOR_DIM = len(embedder.embed_query("test"))
except Exception:
    VECTOR_DIM = 1024

# Crear índices si faltan ----------------------------------------------------
try:
    create_vector_index(
        driver,
        name=vector_index_name,
        label="Chunk",
        embedding_property="embedding",
        dimensions=VECTOR_DIM,
        similarity_fn="cosine",
    )
except Exception:
    pass  # Puede existir

try:
    create_fulltext_index(
        driver,
        name=fulltext_index_name,
        label="Chunk",
        node_properties=["text"],
    )
except Exception:
    pass  # Puede existir

# Índices de propiedades para acelerar búsquedas por nombre ------------------
with driver.session() as _idx_sess:
    _idx_sess.run("CREATE INDEX enzyme_name IF NOT EXISTS FOR (e:Enzyme) ON (e.name)")
    _idx_sess.run(
        "CREATE INDEX metabolite_name IF NOT EXISTS FOR (m:Metabolite) ON (m.name)"
    )
# --------------------------------------------------------------------------- #
# 2) Cypher Retrieval Query
# --------------------------------------------------------------------------- #

RETRIEVAL_QUERY = """
// --- 1) Traverse from retrieved Chunk to Project and related entities --------
MATCH (node)<-[:FROM_CHUNK]-(project:Project)                                    // Each Chunk must belong to a Project
OPTIONAL MATCH (project)-[:IN_REGION]->(region:Region)                           // Project may be in a Region
OPTIONAL MATCH (project)-[:IN_COMMUNE]->(commune:Commune)                        // Project may be in a Commune
OPTIONAL MATCH (project)-[:HAS_PROJECT_TYPE]->(projectType:ProjectType)          // Project may have a ProjectType
OPTIONAL MATCH (project)-[:HAS_TIPOLOGIA]->(tipologia:Tipologia)                 // Project may have a Tipologia

// --- 2) Aggregate lists for multi-valued relationships -----------------------
WITH
  project,                                                                        // Current project node
  collect(DISTINCT region.name) AS regions,                                      // Unique region names (can be multiple)
  collect(DISTINCT commune.name) AS communes,                                    // Unique commune names (can be multiple)
  collect(DISTINCT projectType.name) AS projectTypes,                            // Unique project type names (can be multiple)
  collect(DISTINCT tipologia.code) AS tipologias                                 // Unique tipologia codes (can be multiple)
ORDER BY project.name                                                            // Deterministic ordering of result rows

// --- 3) Return the desired fields --------------------------------------------
RETURN
  project.name AS project_name,                                                  // Project identifier
  project.id AS project_id,                                                      // Project ID if available
  coalesce(regions, []) AS regions,                                              // List of regions (empty list if none)
  coalesce(communes, []) AS communes,                                            // List of communes (empty list if none)
  coalesce(projectTypes, []) AS project_types,                                   // List of project types (empty list if none)
  coalesce(tipologias, []) AS tipologias;                                        // List of tipologia codes (empty list if none)
"""

# --------------------------------------------------------------------------- #
# 3) Configuración HybridCypherRetriever
# --------------------------------------------------------------------------- #

retriever = HybridCypherRetriever(
    driver=driver,
    vector_index_name=vector_index_name,
    fulltext_index_name=fulltext_index_name,
    retrieval_query=RETRIEVAL_QUERY,
    embedder=embedder,
)
