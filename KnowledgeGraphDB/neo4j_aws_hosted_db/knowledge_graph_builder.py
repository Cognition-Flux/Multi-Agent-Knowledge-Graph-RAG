"""Knowledge Graph Builder for AWS Hosted Neo4j Database.

This script creates a Knowledge Graph (KG) in the AWS hosted Neo4j database
from text chunks with proper connection handling and error recovery.

Usage:
    uv run python KnowledgeGraphDB/neo4j_aws_hosted_db/knowledge_graph_builder.py
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar


# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index

from KnowledgeGraphDB.neo4j_aws_hosted_db.connection import (
    close_default_connection,
    get_connection,
)
from src.config import CHUNKS_REFINED_COLLECTION_DIR
from src.documents.markdown_chunking_step02 import load_chunks_from_file


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)

# Connection will be initialized when needed
conn = None


def get_conn():
    """Get or create connection instance."""
    global conn
    if conn is None:
        conn = get_connection()
        if not conn.is_connected():
            # Force connection initialization
            conn.test_connection()
        logger.info(f"✅ Connected to Neo4j at {conn.uri}")
    return conn


def _try_create_embedder() -> Any | None:
    """Create a Bedrock Titan embedder if AWS credentials are available; otherwise return None."""
    import os

    aws_region = os.getenv("AWS_BEDROCK_REGION")
    model_id = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    if not aws_region:
        logger.warning("AWS_BEDROCK_REGION not set. Embeddings will be skipped.")
        return None
    try:
        return BedrockEmbeddings(model_id=model_id, region_name=aws_region)
    except Exception as e:
        logger.warning(f"Failed to create Bedrock embedder: {e}")
        return None


embedder = _try_create_embedder()

# --------------------------------------------------------------------------- #
# Create vector and full-text indexes for Chunk nodes
# --------------------------------------------------------------------------- #

VECTOR_INDEX_NAME = "chunkEmbedding"
FULLTEXT_INDEX_NAME = "chunkFulltext"

# Infer embedding dimensions
if embedder is not None:
    try:
        VECTOR_DIM = len(embedder.embed_query("test"))
        logger.info(f"Embedding dimension: {VECTOR_DIM}")
    except Exception:
        VECTOR_DIM = 1024  # Fallback for Cohere v4
else:
    VECTOR_DIM = 1024  # Default without embedder


def create_indexes():
    """Create necessary indexes in the database."""
    driver = get_conn().driver

    # Create vector index if embedder is available
    if embedder is not None:
        try:
            create_vector_index(
                driver,
                name=VECTOR_INDEX_NAME,
                label="Chunk",
                embedding_property="embedding",
                dimensions=VECTOR_DIM,
                similarity_fn="cosine",
                fail_if_exists=False,
            )
            logger.info(f"✅ Vector index '{VECTOR_INDEX_NAME}' created/verified")
        except Exception as e:
            logger.warning(f"Could not create vector index: {e}")

    # Create full-text index
    try:
        create_fulltext_index(
            driver,
            name=FULLTEXT_INDEX_NAME,
            label="Chunk",
            node_properties=["text"],
            fail_if_exists=False,
        )
        logger.info(f"✅ Full-text index '{FULLTEXT_INDEX_NAME}' created/verified")
    except Exception as e:
        logger.warning(f"Could not create full-text index: {e}")


# --------------------------------------------------------------------------- #
# UUID-based chunk file helpers
# --------------------------------------------------------------------------- #

_UUID_REFINED_JSONL_RE: re.Pattern = re.compile(
    r"^[0-9a-f]{8}_augmented\.jsonl$", re.IGNORECASE
)


def _collect_refined_jsonl_files() -> list[Path]:
    """Return list of refined JSONL files from markdown_chunking_step02."""
    directory = Path(CHUNKS_REFINED_COLLECTION_DIR)
    if not directory.exists():
        raise FileNotFoundError(
            f"Directory {directory} does not exist. Run step02 to generate refined chunks."
        )

    files = sorted(
        p
        for p in directory.glob("*.jsonl")
        if p.is_file() and _UUID_REFINED_JSONL_RE.match(p.name)
    )

    if not files:
        logger.warning(
            "No refined JSONL files matched UUID pattern. Using all *.jsonl files."
        )
        files = sorted(p for p in directory.glob("*.jsonl") if p.is_file())

    return files


def restore_chunks_grouped() -> dict[str, list[Document]]:
    """Load chunks from refined JSONL files grouped by document."""
    grouped: dict[str, list[Document]] = defaultdict(list)
    for jsonl_path in _collect_refined_jsonl_files():
        key = jsonl_path.stem
        if key.endswith("_augmented"):
            key = key[: -len("_augmented")]
        docs = load_chunks_from_file(jsonl_path)
        grouped[key].extend(docs)
    return grouped


# --------------------------------------------------------------------------- #
# Knowledge Graph Schema Definition
# --------------------------------------------------------------------------- #

NODE_TYPES = [
    {
        "label": "Project",
        "description": "Document or project processed in SEA",
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "id", "type": "INTEGER", "required": True},
        ],
    },
    {
        "label": "Chunk",
        "description": "Text fragment from document",
        "properties": [
            {"name": "text", "type": "STRING", "required": True},
            {"name": "chunk_index", "type": "INTEGER"},
            {"name": "h1", "type": "STRING"},
            {"name": "source_path", "type": "STRING"},
            {"name": "uid", "type": "STRING", "required": True},
        ],
    },
    {
        "label": "Region",
        "properties": [{"name": "name", "type": "STRING", "required": True}],
    },
    {
        "label": "Commune",
        "properties": [{"name": "name", "type": "STRING", "required": True}],
    },
    {
        "label": "Tipologia",
        "properties": [{"name": "code", "type": "STRING", "required": True}],
    },
    {
        "label": "ProjectType",
        "properties": [{"name": "name", "type": "STRING", "required": True}],
    },
    {
        "label": "PresentationDate",
        "properties": [{"name": "date", "type": "DATE", "required": True}],
    },
    {
        "label": "DocumentType",
        "properties": [{"name": "name", "type": "STRING", "required": True}],
    },
    {
        "label": "DocumentSubtype",
        "properties": [{"name": "name", "type": "STRING", "required": True}],
    },
]

RELATIONSHIP_TYPES = [
    "HAS_CHUNK",
    "IN_REGION",
    "IN_COMMUNE",
    "HAS_TIPOLOGIA",
    "HAS_PROJECT_TYPE",
    "PRESENTED_ON",
    "HAS_DOCUMENT_TYPE",
    "HAS_DOCUMENT_SUBTYPE",
]


def ensure_property_indexes():
    """Ensure minimal constraints and indexes without destructive operations."""
    connection = get_conn()
    queries = [
        # Unique constraints
        "CREATE CONSTRAINT project_id_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT chunk_uid_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.uid IS UNIQUE",
        # Supporting indexes
        "CREATE INDEX project_name_idx IF NOT EXISTS FOR (p:Project) ON (p.name)",
        "CREATE INDEX region_name_idx IF NOT EXISTS FOR (r:Region) ON (r.name)",
        "CREATE INDEX commune_name_idx IF NOT EXISTS FOR (c:Commune) ON (c.name)",
        "CREATE INDEX tipologia_code_idx IF NOT EXISTS FOR (t:Tipologia) ON (t.code)",
        "CREATE INDEX projecttype_name_idx IF NOT EXISTS FOR (pt:ProjectType) ON (pt.name)",
        "CREATE INDEX doctype_name_idx IF NOT EXISTS FOR (dt:DocumentType) ON (dt.name)",
        "CREATE INDEX docsubtype_name_idx IF NOT EXISTS FOR (ds:DocumentSubtype) ON (ds.name)",
        "CREATE INDEX presentation_date_idx IF NOT EXISTS FOR (d:PresentationDate) ON (d.date)",
    ]

    for query in queries:
        try:
            connection.execute_query(query)
            logger.debug(f"✅ Index/constraint created: {query[:50]}...")
        except Exception as e:
            logger.debug(f"Index/constraint might already exist: {e}")


# --------------------------------------------------------------------------- #
# Utility Functions
# --------------------------------------------------------------------------- #

T = TypeVar("T")


def _with_retry(
    func: Callable[[], T],
    *,
    retries: int = 5,
    base_delay_s: float = 0.5,
    max_delay_s: float = 4.0,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Execute function with exponential backoff retry on recoverable errors."""
    attempt = 0
    while True:
        try:
            return func()
        except retry_exceptions as e:
            attempt += 1
            if attempt > retries:
                raise
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
            logger.warning(f"Retry attempt {attempt}/{retries} after error: {e}")
            time.sleep(delay)


def _compute_chunk_uid(source_path: str, chunk_index: int) -> str:
    """Compute unique identifier for a chunk."""
    raw = f"{source_path}|{chunk_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _safe_int(value: Any, fallback: int = -1) -> int:
    """Safely convert value to integer."""
    try:
        return int(value)
    except Exception:
        return fallback


def _parse_communes(communes_value: str | list[str] | None) -> list[str]:
    """Parse and normalize commune field into a list."""
    if communes_value is None:
        return []
    if isinstance(communes_value, list):
        return [c.strip() for c in communes_value if isinstance(c, str) and c.strip()]
    s = str(communes_value)
    parts = re.split(r"[,/\-–—]+", s)
    return [p.strip() for p in parts if p.strip()]


def _iso_date_to_map(date_str: str | None) -> dict[str, int] | None:
    """Convert 'YYYY-MM-DD' to a map {year, month, day} for Cypher date()."""
    if not date_str or not isinstance(date_str, str):
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", date_str)
    if not m:
        return None
    year, month, day = m.groups()
    try:
        return {"year": int(year), "month": int(month), "day": int(day)}
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Main Knowledge Graph Building Function
# --------------------------------------------------------------------------- #


async def build_kg_from_docs(docs: list[Document]) -> None:
    """Build the knowledge graph from the list of documents."""
    ensure_property_indexes()

    # Initialize counters for summary
    summary_counts = {"chunks": 0, "presentation_date": 0, "skipped": 0}
    unique_regions: set[str] = set()
    unique_communes: set[str] = set()
    unique_tipologias: set[str] = set()
    unique_project_types: set[str] = set()
    seen_project_ids: set[int] = set()

    total_chunks = len(docs)

    # Calculate total chunks per document
    doc_total_chunks: dict[str, int] = {}
    for d in docs:
        meta_d = d.metadata  # type: ignore[attr-defined]
        spath = str(meta_d.get("source_path"))
        cidx = int(meta_d.get("chunk_index", 0))
        doc_total_chunks[spath] = max(doc_total_chunks.get(spath, -1), cidx)
    for spath in list(doc_total_chunks.keys()):
        doc_total_chunks[spath] = doc_total_chunks[spath] + 1

    # Get unique projects and documents in batch
    unique_projects_in_batch: set[int] = set()
    unique_docs_in_batch: set[str] = set()
    for d in docs:
        m = d.metadata  # type: ignore[attr-defined]
        try:
            unique_projects_in_batch.add(int(m.get("id")))
        except Exception:
            pass
        spath = m.get("source_path")
        if spath:
            unique_docs_in_batch.add(str(spath))

    logger.info(
        f"Starting KG load → projects: {len(unique_projects_in_batch)}, "
        f"documents: {len(unique_docs_in_batch)}, chunks: {total_chunks}"
    )

    for doc in docs:
        meta = doc.metadata  # type: ignore[attr-defined]

        # Extract metadata fields
        region = meta.get("region")
        communes_raw = meta.get("ei_document_communes") or []
        communes_list = _parse_communes(communes_raw)
        tipologia = meta.get("tipologia")
        project_type = meta.get("tipo_de_proyecto")
        project_name = meta.get("nombre")
        project_id_int = _safe_int(meta.get("id"), -1)
        source_path_val = meta.get("source_path")

        if not source_path_val or not isinstance(source_path_val, str):
            logger.warning("⚠️  Chunk without valid 'source_path'. Skipping.")
            summary_counts["skipped"] += 1
            continue

        source_path = source_path_val
        chunk_idx = _safe_int(meta.get("chunk_index", 0), 0)
        total_in_doc = doc_total_chunks.get(source_path, 0)
        chunk_uid = _compute_chunk_uid(source_path, chunk_idx)

        # Progress logging
        logger.info(
            f"[{summary_counts['chunks'] + 1}/{total_chunks}] "
            f"Project {project_id_int} - {project_name}"
        )
        logger.info(
            f"  Doc: {Path(source_path).name} chunk {chunk_idx + 1}/{total_in_doc}"
        )

        # Update unique sets
        if region:
            unique_regions.add(region)
        unique_communes.update(communes_list)
        if tipologia:
            unique_tipologias.add(tipologia)
        if project_type:
            unique_project_types.add(project_type)

        # 1) Create or verify Chunk exists
        def _ensure_chunk(
            uid: str = chunk_uid,
            page_text: str = doc.page_content,
            idx: int = chunk_idx,
            src: str = source_path,
            heading: str | None = meta.get("h1"),
        ) -> bool:
            """Create chunk if it doesn't exist. Returns True if created."""
            # Check if chunk exists
            records, _, _ = conn.execute_query(
                """
                MATCH (c:Chunk)
                WHERE c.uid = $uid
                   OR (c.source_path = $src AND c.chunk_index = $idx)
                RETURN c LIMIT 1
                """,
                {"uid": uid, "src": src, "idx": idx},
            )

            if records:
                # Ensure uid is set
                conn.execute_query(
                    """
                    MATCH (c:Chunk)
                    WHERE id(c) = $nid
                    SET c.uid = coalesce(c.uid, $uid)
                    """,
                    {"nid": records[0]["c"].id, "uid": uid},
                )
                return False

            # Create new chunk
            conn.execute_query(
                """
                CREATE (c:Chunk {
                    uid: $uid,
                    text: $text,
                    chunk_index: $chunk_index,
                    source_path: $source_path,
                    h1: $h1
                })
                """,
                {
                    "uid": uid,
                    "text": page_text,
                    "chunk_index": idx,
                    "source_path": src,
                    "h1": heading,
                },
            )
            return True

        created = _with_retry(_ensure_chunk)

        if not created:
            logger.debug(f"Chunk already exists (uid={chunk_uid}). Skipping.")
            summary_counts["skipped"] += 1
            continue

        # 2) Add embedding if available
        if embedder is not None and created:
            try:
                embedding_vector = embedder.embed_query(doc.page_content)

                def _set_embedding(
                    uid: str = chunk_uid, emb: list[float] = embedding_vector
                ) -> None:
                    conn.execute_query(
                        "MATCH (c:Chunk {uid: $uid}) SET c.embedding = $embedding",
                        {"uid": uid, "embedding": emb},
                    )

                _with_retry(_set_embedding)
                logger.debug(f"✅ Embedding added for chunk {chunk_uid}")
            except Exception as e:
                logger.warning(f"Failed to add embedding: {e}")

        # Parse presentation date
        presentation_date_str = meta.get("fecha_de_presentacion")
        presentation_date_map = _iso_date_to_map(presentation_date_str)

        # 3) Create Project and relationships
        if project_id_int != -1:

            def _ensure_project_and_links(
                uid: str = chunk_uid,
                pid: int = project_id_int,
                pname: str | None = project_name,
                pres_date: dict[str, int] | None = presentation_date_map,
                reg: str | None = region,
                communes: list[str] = communes_list,
                tip: str | None = tipologia,
                ptype: str | None = project_type,
                doctype: str | None = meta.get("type"),
                docsub: str | None = meta.get("subtype"),
            ) -> None:
                """Create project and all its relationships."""
                # Project and HAS_CHUNK relationship
                conn.execute_query(
                    """
                    MATCH (c:Chunk {uid: $uid})
                    MERGE (p:Project {id: $project_id})
                    ON CREATE SET p.name = $project_name
                    MERGE (p)-[:HAS_CHUNK]->(c)
                    """,
                    {"uid": uid, "project_id": pid, "project_name": pname},
                )

                # Presentation date
                if pres_date:
                    conn.execute_query(
                        """
                        MATCH (p:Project {id: $project_id})
                        MERGE (d:PresentationDate {date: date($fecha)})
                        MERGE (p)-[:PRESENTED_ON]->(d)
                        """,
                        {"project_id": pid, "fecha": pres_date},
                    )

                # Region
                if reg:
                    conn.execute_query(
                        """
                        MATCH (p:Project {id: $project_id})
                        MERGE (r:Region {name: $region})
                        MERGE (p)-[:IN_REGION]->(r)
                        """,
                        {"project_id": pid, "region": reg},
                    )

                # Communes
                for commune in communes:
                    conn.execute_query(
                        """
                        MATCH (p:Project {id: $project_id})
                        MERGE (c:Commune {name: $commune})
                        MERGE (p)-[:IN_COMMUNE]->(c)
                        """,
                        {"project_id": pid, "commune": commune},
                    )

                # Tipologia
                if tip:
                    conn.execute_query(
                        """
                        MATCH (p:Project {id: $project_id})
                        MERGE (t:Tipologia {code: $tipologia})
                        MERGE (p)-[:HAS_TIPOLOGIA]->(t)
                        """,
                        {"project_id": pid, "tipologia": tip},
                    )

                # Project type
                if ptype:
                    conn.execute_query(
                        """
                        MATCH (p:Project {id: $project_id})
                        MERGE (pt:ProjectType {name: $project_type})
                        MERGE (p)-[:HAS_PROJECT_TYPE]->(pt)
                        """,
                        {"project_id": pid, "project_type": ptype},
                    )

                # Document type
                if doctype:
                    conn.execute_query(
                        """
                        MATCH (p:Project {id: $project_id})
                        MERGE (dt:DocumentType {name: $doc_type})
                        MERGE (p)-[:HAS_DOCUMENT_TYPE]->(dt)
                        """,
                        {"project_id": pid, "doc_type": doctype},
                    )

                # Document subtype
                if docsub:
                    conn.execute_query(
                        """
                        MATCH (p:Project {id: $project_id})
                        MERGE (ds:DocumentSubtype {name: $doc_subtype})
                        MERGE (p)-[:HAS_DOCUMENT_SUBTYPE]->(ds)
                        """,
                        {"project_id": pid, "doc_subtype": docsub},
                    )

            _with_retry(_ensure_project_and_links)

            if presentation_date_map:
                summary_counts["presentation_date"] += 1

        # Log project details (first time only)
        if project_id_int != -1 and project_id_int not in seen_project_ids:
            logger.info(
                f"  Detail → Region='{region}', Tipologia='{tipologia}', "
                f"Type='{project_type}', Subtype='{meta.get('subtype')}', "
                f"Communes={communes_list}"
            )
            seen_project_ids.add(project_id_int)

        summary_counts["chunks"] += 1
        logger.info(f"✅ Project {meta.get('id')} ready.")

    # Print final summary
    logger.info("🎉 Knowledge graph creation finished!")
    logger.info("------ SUMMARY ------")
    logger.info(f"Chunks processed: {summary_counts['chunks']}")
    logger.info(f"Chunks skipped: {summary_counts['skipped']}")
    logger.info(f"Presentation dates linked: {summary_counts['presentation_date']}")
    logger.info(f"Unique regions: {len(unique_regions)}")
    logger.info(f"Unique communes: {len(unique_communes)}")
    logger.info(f"Unique tipologias: {len(unique_tipologias)}")
    logger.info(f"Unique project types: {len(unique_project_types)}")
    logger.info("----------------------")


# --------------------------------------------------------------------------- #
# Entry Point
# --------------------------------------------------------------------------- #


async def main():
    """Main entry point for the knowledge graph builder."""
    try:
        # Create indexes
        create_indexes()

        # Load chunks
        logger.info("Loading chunks from refined collection...")
        chunk_dict = restore_chunks_grouped()
        total_chunks = sum(len(v) for v in chunk_dict.values())
        logger.info(f"✓ Chunks loaded: {total_chunks} in {len(chunk_dict)} documents.")

        # Concatenate all chunks
        all_docs: list[Document] = []
        for _k, _docs in chunk_dict.items():
            all_docs.extend(_docs)

        # Build knowledge graph
        await build_kg_from_docs(all_docs)

    except Exception as e:
        logger.error(f"Failed to build knowledge graph: {e}")
        raise
    finally:
        # Close connection
        close_default_connection()
        logger.info("Connection closed.")


if __name__ == "__main__":
    asyncio.run(main())
