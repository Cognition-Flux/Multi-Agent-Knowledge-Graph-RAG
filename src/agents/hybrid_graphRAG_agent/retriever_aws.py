"""Retriever module for Hybrid GraphRAG Agent using AWS-hosted Neo4j.

This module creates a hybrid retriever that combines full-text and vector searches
using the AWS-hosted Neo4j database.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index
from neo4j_graphrag.retrievers import HybridCypherRetriever

from KnowledgeGraphDB.neo4j_aws_hosted_db.connection import get_connection


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Bedrock Embeddings Wrapper for neo4j_graphrag compatibility
# --------------------------------------------------------------------------- #


class BedrockEmbedderAdapter(Embedder):
    """Adapter to make BedrockEmbeddings compatible with neo4j_graphrag."""

    def __init__(self, bedrock_embeddings: BedrockEmbeddings):
        """Initialize with a BedrockEmbeddings instance."""
        self.bedrock_embeddings = bedrock_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return self.bedrock_embeddings.embed_query(text)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        return self.bedrock_embeddings.embed_documents(texts)


# --------------------------------------------------------------------------- #
# 1) Get AWS Neo4j connection
# --------------------------------------------------------------------------- #

# Get the connection instance
connection = get_connection()
driver = connection.driver

# Verify connectivity
try:
    connection.test_connection()
    logger.info("✅ Connected to AWS-hosted Neo4j database")
except Exception as e:
    logger.error(f"❌ Failed to connect to AWS Neo4j: {e}")
    raise

# --------------------------------------------------------------------------- #
# 2) Configure embeddings
# --------------------------------------------------------------------------- #

# Configure AWS Bedrock Embeddings
model_id = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
aws_region = os.getenv("AWS_BEDROCK_REGION", "us-west-2")

logger.info(f"Configuring Bedrock Embeddings with model: {model_id}")

# Create BedrockEmbeddings instance
bedrock_embeddings = BedrockEmbeddings(
    model_id=model_id,
    region_name=aws_region,
)

# Wrap for neo4j_graphrag compatibility
embedder = BedrockEmbedderAdapter(bedrock_embeddings)

# Index names for :Chunk nodes created by SimpleKGPipeline
vector_index_name = "chunkEmbedding"
fulltext_index_name = "chunkFulltext"

# Infer dimensionality dynamically
try:
    test_embedding = embedder.embed_query("test")
    VECTOR_DIM = len(test_embedding)
    logger.info(f"Detected embedding dimension: {VECTOR_DIM}")
except Exception as e:
    logger.warning(f"Could not infer embedding dimension: {e}. Using default 1024")
    VECTOR_DIM = 1024

# --------------------------------------------------------------------------- #
# 3) Create indexes if missing
# --------------------------------------------------------------------------- #

# Vector index
try:
    create_vector_index(
        driver,
        name=vector_index_name,
        label="Chunk",
        embedding_property="embedding",
        dimensions=VECTOR_DIM,
        similarity_fn="cosine",
    )
    logger.info(f"✅ Vector index '{vector_index_name}' created or already exists")
except Exception as e:
    logger.debug(f"Vector index creation note: {e}")

# Full-text index
try:
    create_fulltext_index(
        driver,
        name=fulltext_index_name,
        label="Chunk",
        node_properties=["text"],
    )
    logger.info(f"✅ Full-text index '{fulltext_index_name}' created or already exists")
except Exception as e:
    logger.debug(f"Full-text index creation note: {e}")

# Property indexes for faster searches
with driver.session() as idx_session:
    try:
        idx_session.run(
            "CREATE INDEX enzyme_name IF NOT EXISTS FOR (e:Enzyme) ON (e.name)"
        )
        idx_session.run(
            "CREATE INDEX metabolite_name IF NOT EXISTS FOR (m:Metabolite) ON (m.name)"
        )
        logger.info("✅ Property indexes created or already exist")
    except Exception as e:
        logger.debug(f"Property index creation note: {e}")

# --------------------------------------------------------------------------- #
# 4) Cypher Retrieval Query
# --------------------------------------------------------------------------- #

# Enriched query with complete project and chunk context
RETRIEVAL_QUERY = """
// 1. Capture information from retrieved chunk and its project
MATCH (node)<-[:HAS_CHUNK]-(project:Project)
WITH DISTINCT project, node

// 2. Get geographic metadata
OPTIONAL MATCH (project)-[:IN_REGION]->(region:Region)
OPTIONAL MATCH (project)-[:IN_COMMUNE]->(commune:Commune)

// 3. Get project classification metadata
OPTIONAL MATCH (project)-[:HAS_PROJECT_TYPE]->(projectType:ProjectType)
OPTIONAL MATCH (project)-[:HAS_TIPOLOGIA]->(tipologia:Tipologia)

// 4. Get temporal metadata
OPTIONAL MATCH (project)-[:PRESENTED_ON]->(presentationDate:PresentationDate)

// 5. Get document metadata
OPTIONAL MATCH (project)-[:HAS_DOCUMENT_TYPE]->(docType:DocumentType)
OPTIONAL MATCH (project)-[:HAS_DOCUMENT_SUBTYPE]->(docSubtype:DocumentSubtype)

// 6. Count related chunks for project size context
OPTIONAL MATCH (project)-[:HAS_CHUNK]->(allChunks:Chunk)

// 7. Aggregate all collected information
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

// 8. Build structured response with enriched metadata
RETURN
  // Basic project information
  coalesce(project.name, 'Proyecto sin nombre') AS project_name,
  coalesce(project.id, -1) AS project_id,

  // Specific chunk context
  node.chunk_index AS chunk_index,
  coalesce(node.h1, 'Sin título de sección') AS section_title,
  coalesce(node.source_path, 'Ruta no disponible') AS source_document,
  substring(node.text, 0, 500) AS chunk_preview,

  // Geographic information
  CASE
    WHEN size(regions) > 0 THEN regions
    ELSE ['Sin región especificada']
  END AS regions,
  CASE
    WHEN size(communes) > 0 THEN communes
    ELSE ['Sin comuna especificada']
  END AS communes,
  size(communes) AS num_communes,

  // Project classification
  CASE
    WHEN size(project_types) > 0 THEN project_types
    ELSE ['Tipo de proyecto no especificado']
  END AS project_types,
  CASE
    WHEN size(tipologia_codes) > 0 THEN tipologia_codes
    ELSE ['Sin código de tipología']
  END AS tipologia_codes,

  // Document information
  CASE
    WHEN size(document_types) > 0 THEN document_types
    ELSE ['Tipo de documento no especificado']
  END AS document_types,
  CASE
    WHEN size(document_subtypes) > 0 THEN document_subtypes
    ELSE ['Subtipo no especificado']
  END AS document_subtypes,

  // Temporal information
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

  // Additional context metadata
  total_chunks_in_project AS project_size_in_chunks,

  // Geographic summary for better context
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
# 5) Configure HybridCypherRetriever
# --------------------------------------------------------------------------- #

retriever = HybridCypherRetriever(
    driver=driver,
    vector_index_name=vector_index_name,
    fulltext_index_name=fulltext_index_name,
    retrieval_query=RETRIEVAL_QUERY,
    embedder=embedder,
)

logger.info("✅ HybridCypherRetriever configured with AWS Neo4j")

# --------------------------------------------------------------------------- #
# 6) Test the retriever (only runs if this is the main script)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("\n🔍 Testing AWS Neo4j Hybrid Retriever with Bedrock Embeddings")
    print("=" * 50)

    # Test connection
    test_results = connection.test_connection()
    print(f"📊 Database: {test_results['database']}")
    print(f"📊 Connected: {test_results['connected']}")
    if test_results.get("server_version"):
        print(f"📊 Server Version: {test_results['server_version']}")

    # Test embeddings
    print("\n📊 Embedding Configuration:")
    print(f"   Model: {model_id}")
    print(f"   Region: {aws_region}")
    print(f"   Dimension: {VECTOR_DIM}")

    # Test embedding generation
    try:
        test_text = "test embedding"
        test_emb = embedder.embed_query(test_text)
        print(f"✅ Embedding test successful (dimension: {len(test_emb)})")
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")

    # Test query
    QUERY = "¿Qué información tienes sobre proyectos de biosólidos?"
    print(f"\n🔍 Test Query: {QUERY}")

    try:
        # Test retriever configuration
        print("✅ Retriever configured successfully with Bedrock embeddings")

        # Note: Full GraphRAG test requires LLM configuration
        print("\n📝 Note: Full GraphRAG search requires LLM configuration.")
        print("   The retriever is ready to use with Bedrock embeddings.")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
