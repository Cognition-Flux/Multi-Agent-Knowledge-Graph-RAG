"""Este script crea un retriever híbrido que combina búsquedas en texto completo y búsquedas vectoriales."""

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

# Consulta enriquecida con contexto completo del proyecto y chunk
RETRIEVAL_QUERY = """
// 1. Capturar información del chunk recuperado y su proyecto
MATCH (node)<-[:HAS_CHUNK]-(project:Project)
WITH DISTINCT project, node

// 2. Obtener metadatos geográficos
OPTIONAL MATCH (project)-[:IN_REGION]->(region:Region)
OPTIONAL MATCH (project)-[:IN_COMMUNE]->(commune:Commune)

// 3. Obtener metadatos de clasificación del proyecto
OPTIONAL MATCH (project)-[:HAS_PROJECT_TYPE]->(projectType:ProjectType)
OPTIONAL MATCH (project)-[:HAS_TIPOLOGIA]->(tipologia:Tipologia)

// 4. Obtener metadatos temporales
OPTIONAL MATCH (project)-[:PRESENTED_ON]->(presentationDate:PresentationDate)

// 5. Obtener metadatos documentales
OPTIONAL MATCH (project)-[:HAS_DOCUMENT_TYPE]->(docType:DocumentType)
OPTIONAL MATCH (project)-[:HAS_DOCUMENT_SUBTYPE]->(docSubtype:DocumentSubtype)

// 6. Contar chunks relacionados para contexto de tamaño del proyecto
OPTIONAL MATCH (project)-[:HAS_CHUNK]->(allChunks:Chunk)

// 7. Agregar toda la información recolectada
WITH project,
     node,
     collect(DISTINCT region.name) AS regions,
     collect(DISTINCT commune.name) AS communes,
     collect(DISTINCT projectType.name) AS project_types,
     collect(DISTINCT tipologia.code) AS tipologia_codes,
     collect(DISTINCT docType.name) AS document_types,
     collect(DISTINCT docSubtype.name) AS document_subtypes,
     collect(DISTINCT presentationDate.date) AS presentation_dates,
     count(DISTINCT allChunks) AS total_chunks_in_project,
     collect(DISTINCT {
         region: region.name,
         communes: commune.name
     }) AS geographic_context

// 8. Construir respuesta estructurada con metadatos enriquecidos
RETURN
  // Información básica del proyecto
  coalesce(project.name, 'Proyecto sin nombre') AS project_name,
  coalesce(project.id, -1) AS project_id,

  // Contexto del chunk específico
  node.chunk_index AS chunk_index,
  coalesce(node.h1, 'Sin título de sección') AS section_title,
  coalesce(node.source_path, 'Ruta no disponible') AS source_document,
  substring(node.text, 0, 500) AS chunk_preview,

  // Información geográfica
  CASE
    WHEN size(regions) > 0 THEN regions
    ELSE ['Sin región especificada']
  END AS regions,
  CASE
    WHEN size(communes) > 0 THEN communes
    ELSE ['Sin comuna especificada']
  END AS communes,
  size(communes) AS num_communes,

  // Clasificación del proyecto
  CASE
    WHEN size(project_types) > 0 THEN project_types
    ELSE ['Tipo de proyecto no especificado']
  END AS project_types,
  CASE
    WHEN size(tipologia_codes) > 0 THEN tipologia_codes
    ELSE ['Sin código de tipología']
  END AS tipologia_codes,

  // Información documental
  CASE
    WHEN size(document_types) > 0 THEN document_types
    ELSE ['Tipo de documento no especificado']
  END AS document_types,
  CASE
    WHEN size(document_subtypes) > 0 THEN document_subtypes
    ELSE ['Subtipo no especificado']
  END AS document_subtypes,

  // Información temporal
  CASE
    WHEN size(presentation_dates) > 0 THEN
      [d IN presentation_dates | toString(d)]
    ELSE ['Fecha de presentación no disponible']
  END AS presentation_dates,
  CASE
    WHEN size(presentation_dates) > 0 THEN
      toString(presentation_dates[0])
    ELSE 'Sin fecha'
  END AS first_presentation_date,

  // Metadatos adicionales de contexto
  total_chunks_in_project AS project_size_in_chunks,

  // Resumen geográfico concatenado para mejor contexto
  CASE
    WHEN size(regions) > 0 AND size(communes) > 0 THEN
      regions[0] + ' - ' + reduce(s = '', c IN communes[0..3] |
        CASE WHEN s = '' THEN c ELSE s + ', ' + c END)
    WHEN size(regions) > 0 THEN regions[0]
    WHEN size(communes) > 0 THEN 'Comuna(s): ' + communes[0]
    ELSE 'Ubicación no especificada'
  END AS geographic_summary

ORDER BY project.id, node.chunk_index
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

# --------------------------------------------------------------------------- #
# 4) Test del retriever (solo se ejecuta si es el script principal)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from neo4j_graphrag.generation import GraphRAG, RagTemplate
    from neo4j_graphrag.llm import AzureOpenAILLM

    # Configurar LLM
    llm = AzureOpenAILLM(
        model_name="gpt-4.1",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    # Template para RAG
    rag_template = RagTemplate(
        template="""You are an expert in the projects. Answer the **Question** ONLY
using the **Context** provided.

NEVER add NOR inject information or data that is not in the context.

# Question:
{query_text}

# Context:
{context}

# Answer:
""",
        expected_inputs=["query_text", "context"],
    )

    # Crear GraphRAG con el retriever local
    graph_rag = GraphRAG(retriever=retriever, llm=llm, prompt_template=rag_template)

    # Ejecutar prueba
    QUERY = "que informacion tienes de CENTRO DE RECEPCIÓN Y DISPOSICIÓN FINAL DE BIOSÓLIDOS"
    print(f"\n🔍 Consultando: {QUERY}\n")

    try:
        response = graph_rag.search(
            QUERY,
            retriever_config={"top_k": 100},
            return_context=True,
        )
        print("✅ Respuesta:")
        print(response.answer)
    except Exception as e:
        print(f"❌ Error: {e}")
