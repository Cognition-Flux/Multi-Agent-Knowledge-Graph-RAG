"""Project-wide configuration constants."""

from pathlib import Path


# AWS S3 bucket where crawler documents are stored
BUCKET_NAME: str = "nviro-crawlers"

# Base directory for local document collections
BASE_DIR = Path(__file__).resolve().parent

# Directory where PDF files will be stored locally
PDF_COLLECTION_DIR = BASE_DIR / "documents" / "collections" / "pdf"

# Directory where extracted markdown files will be saved
MARKDOWN_RAW_COLLECTION_DIR = (
    BASE_DIR / "documents" / "collections" / "markdown" / "raw"
)
MARKDOWN_REFINED_COLLECTION_DIR = (
    BASE_DIR / "documents" / "collections" / "markdown" / "refined"
)

# Path to flora & fauna metadata Parquet file
# FLORA_FAUNA_PARQUET_PATH = BASE_DIR / "cli" / "flora_fauna_metadata.parquet"
FLORA_FAUNA_PARQUET_PATH = BASE_DIR / "cli" / "flora_fauna_metadata_round_2.parquet"


# Directory where JSONL chunks will be stored (raw version)
CHUNKS_RAW_COLLECTION_DIR = (
    BASE_DIR / "documents" / "collections" / "chunks" / "raw_chunks"
)

# Directory where JSONL chunks with enriched metadata will be stored
CHUNKS_REFINED_COLLECTION_DIR = (
    BASE_DIR / "documents" / "collections" / "chunks" / "refined_chunks"
)

# ---------------------------------------------------------------------------
# PDF extraction backend toggle
# True  -> usa AzureAIDocumentIntelligenceLoader (SaaS)
# False -> usa LocalPDFMarkdownLoader definido en localPDFparse.parse
# ---------------------------------------------------------------------------

USE_SAAS_PDF_PARSER: bool = False
