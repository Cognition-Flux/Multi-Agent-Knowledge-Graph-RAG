"""Re-embed all Chunk nodes with AWS Titan and recreate the vector index.

Usage:
    uv run -m src.tools.reembed_chunks_bedrock_titan

This script:
  1) Connects to the AWS-hosted Neo4j DB using the standard connection module
  2) Drops any existing vector index on Chunk(embedding)
  3) Re-embeds all Chunk.text with Bedrock Titan and updates c.embedding
  4) Recreates the vector index with the correct dimension (1024 for Titan v2)
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Iterable

from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from neo4j import Driver
from neo4j_graphrag.indexes import create_vector_index

from KnowledgeGraphDB.neo4j_aws_hosted_db.connection import (
    close_default_connection,
    get_connection,
)


load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


VECTOR_INDEX_NAME = os.getenv("KG_VECTOR_INDEX_NAME", "chunkEmbedding")
EMBED_PROP = os.getenv("KG_EMBED_PROP", "embedding")
LABEL = os.getenv("KG_CHUNK_LABEL", "Chunk")


def _drop_vector_index_if_exists(driver: Driver) -> None:
    """Drop the target vector index by name using Neo4j 5 syntax."""
    with driver.session() as session:
        log.info("Dropping index '%s' if it exists", VECTOR_INDEX_NAME)
        session.run(f"DROP INDEX `{VECTOR_INDEX_NAME}` IF EXISTS")


def _iterate_chunks(
    driver: Driver, batch_size: int = 200
) -> Iterable[list[tuple[str, str]]]:
    skip = 0
    while True:
        with driver.session() as session:
            records = session.run(
                (
                    "MATCH (c:"
                    + LABEL
                    + ") RETURN c.uid AS uid, c.text AS text ORDER BY c.uid SKIP $skip LIMIT $limit"
                ),
                {"skip": skip, "limit": batch_size},
            ).values()
        if not records:
            break
        yield [(uid or "", text or "") for uid, text in records]
        skip += batch_size


def _set_embedding_batch(driver: Driver, items: list[tuple[str, list[float]]]) -> None:
    with driver.session() as session:
        for uid, emb in items:
            session.run(
                "MATCH (c:" + LABEL + " {uid: $uid}) SET c." + EMBED_PROP + " = $emb",
                {"uid": uid, "emb": emb},
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-embed Chunk nodes and manage vector index"
    )
    parser.add_argument(
        "--mode",
        choices=["index-only", "embed-only", "both"],
        default="both",
        help="Operation mode: just rebuild index, just embed, or both",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding updates (embed-only/both modes)",
    )
    args = parser.parse_args()

    conn = get_connection()
    driver = conn.driver

    # Configure Titan embeddings
    model_id = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
    region = os.getenv("AWS_BEDROCK_REGION", "us-west-2")
    embedder = BedrockEmbeddings(model_id=model_id, region_name=region)

    # Verify output dimension
    dim = len(embedder.embed_query("dimension-check"))
    log.info("Titan embedding dimension detected: %s", dim)

    total = 0

    if args.mode in ("index-only", "both"):
        _drop_vector_index_if_exists(driver)
        log.info(
            "Creating vector index '%s' on %s(%s) with dim=%d",
            VECTOR_INDEX_NAME,
            LABEL,
            EMBED_PROP,
            dim,
        )
        create_vector_index(
            driver,
            name=VECTOR_INDEX_NAME,
            label=LABEL,
            embedding_property=EMBED_PROP,
            dimensions=dim,
            similarity_fn="cosine",
            fail_if_exists=False,
        )

    if args.mode in ("embed-only", "both"):
        # Re-embed all chunks in batches
        for batch in _iterate_chunks(driver, batch_size=max(1, args.batch_size)):
            if not batch:
                break
            texts = [t for _, t in batch]
            uids = [u for u, _ in batch]
            vectors = embedder.embed_documents(texts)
            _set_embedding_batch(driver, list(zip(uids, vectors, strict=False)))
            total += len(batch)
            log.info("Re-embedded %d chunks (cumulative)", total)

    close_default_connection()
    if args.mode == "index-only":
        log.info("Done. Recreated vector index only (dim=%d).", dim)
    elif args.mode == "embed-only":
        log.info("Done. Re-embedded %d chunks.", total)
    else:
        log.info("Done. Recreated index and re-embedded %d chunks.", total)


if __name__ == "__main__":
    main()
