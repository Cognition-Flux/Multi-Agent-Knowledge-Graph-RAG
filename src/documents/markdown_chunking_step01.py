# %%
"""Chunk refined Markdown documents for vector storage.

This utility reads every Markdown file located in the directory specified by
`MARKDOWN_REFINED_COLLECTION_DIR` (see `src.config`).  Each document is first
split by Markdown headings (``#``, ``##``, ``###``) to preserve the inherent
hierarchy.  Then, any long section is further broken down with
`RecursiveCharacterTextSplitter` so that chunks stay below `chunk_size` while
optionally overlapping to maintain context across boundaries.

The result is a list of `langchain_core.documents.Document` objects ready to be
inserted into a vector store for Retrieval-Augmented Generation (RAG).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.config import CHUNKS_RAW_COLLECTION_DIR, MARKDOWN_REFINED_COLLECTION_DIR


# Initialise environment variables (for local debugging if needed)
load_dotenv(override=True)

###############################################################################
# Helpers                                                                     #
###############################################################################

# Prefix used by the second ingestion round to uniquely identify files
_MD_PREFIX = "round_2_"

# ---------------------------------------------------------------------------
# Naming helpers                                                              #
# ---------------------------------------------------------------------------

# Regex to capture the 8-char hexadecimal UUID used in round_2 filenames
_UUID_REGEX: re.Pattern = re.compile(r"^round_2_([0-9a-f]{8})_", re.IGNORECASE)


def _extract_uuid(file_name: str) -> str | None:
    """Return the hexadecimal UUID captured from *file_name* or *None*."""
    m = _UUID_REGEX.match(file_name)
    if m:
        return m.group(1)
    return None


def collect_markdown_files() -> list[Path]:
    """Return a list of refined Markdown paths belonging to the current ingestion round.

    The parsing pipeline stores every file with a unique name that starts with
    the ``round_2_`` prefix (e.g., ``round_2_8fae1c34_myfile_gpt-4.1.md``).  To
    avoid mixing previous ingestion rounds, we initially restrict the search
    to that prefix.  If **no** files are found (e.g., when running on an
    earlier dataset), we gracefully fall back to *all* ``*.md`` files while
    logging a warning so users are aware of the situation.
    """
    directory = Path(MARKDOWN_REFINED_COLLECTION_DIR)
    if not directory.exists():
        raise FileNotFoundError(
            f"Directory {directory} does not exist. "
            "Make sure the refined Markdown collection has been generated."
        )

    # First, look for Markdown files that follow the expected naming scheme
    files = sorted(
        p
        for p in directory.glob("*.md")
        if p.is_file() and p.name.startswith(_MD_PREFIX)
    )

    if not files:
        # No files with the prefix were found – keep backward-compatibility
        print(
            f"! WARNING: No markdown files found with prefix '{_MD_PREFIX}'. "
            "Falling back to every *.md file in the directory."
        )
        files = sorted(p for p in directory.glob("*.md") if p.is_file())

    return files


###############################################################################
# Chunking logic                                                              #
###############################################################################


# Headings to split on (level → metadata key)
_HEADINGS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


# Re-usable splitter instances ------------------------------------------------
_HEADER_SPLITTER = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADINGS_TO_SPLIT_ON)
_CHAR_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1024,  # characters – tweak as needed
    chunk_overlap=128,  # keep some overlap for context
)

# ---------------------------------------------------------------------------
# Statistics collection for reporting
# ---------------------------------------------------------------------------
_FILE_CHUNK_INFO: list[tuple[str, int]] = []


def _chunk_single_markdown(text: str, source_path: Path) -> Sequence[Document]:
    """Split *text* into quality chunks preserving header metadata.

    Args:
        text:      Full Markdown string.
        source_path: Path to the original Markdown file (added to metadata).

    Returns:
        A sequence of `Document` objects with ``page_content`` ≤ ``chunk_size``
        and rich metadata (heading hierarchy, source, chunk index).
    """
    # Extract UUID from the file name once and attach it to every chunk's metadata
    doc_uuid: str | None = _extract_uuid(source_path.name)
    if doc_uuid is None:
        print(f"! WARNING: Could not extract UUID from {source_path.name}")
        doc_uuid = "unknown"

    header_docs = _HEADER_SPLITTER.split_text(text)

    chunks: list[Document] = []
    for doc in header_docs:
        # Further split if this header section is still too large
        sub_texts = _CHAR_SPLITTER.split_text(doc.page_content)
        for idx, chunk_text in enumerate(sub_texts):
            metadata = {
                **doc.metadata,  # heading hierarchy (e.g., {"h1": ..., "h2": ...})
                "source_path": str(source_path),
                "doc_uuid": doc_uuid,
                "chunk_index": idx,
            }
            chunks.append(Document(page_content=chunk_text, metadata=metadata))
    return chunks


def chunk_all_markdown_files() -> list[Document]:
    """Chunk every Markdown file inside *MARKDOWN_REFINED_COLLECTION_DIR*."""
    all_chunks: list[Document] = []
    for md_path in collect_markdown_files():
        with open(md_path, encoding="utf-8") as fh:
            text = fh.read()
        file_chunks = _chunk_single_markdown(text, md_path)
        _FILE_CHUNK_INFO.append((md_path.name, len(file_chunks)))
        all_chunks.extend(file_chunks)
        print(f"✓ {md_path.name:<50} → {len(file_chunks):>4} chunks")

    print("\nTotal chunks generated:", len(all_chunks))
    return all_chunks


###############################################################################
# CLI-entry point                                                             #
###############################################################################
chunks = chunk_all_markdown_files()

###############################################################################
# Persist chunks to disk                                                      #
###############################################################################

import json
from collections import defaultdict


def _group_chunks_by_source(docs: Sequence[Document]):
    """Group *docs* by their extracted UUID so each original PDF maps to one file."""
    grouped: dict[str, list[Document]] = defaultdict(list)
    for doc in docs:
        uuid = doc.metadata.get("doc_uuid")
        if not uuid:
            # Fallback extraction if somehow missing
            uuid = (
                _extract_uuid(Path(doc.metadata.get("source_path", "")).name)
                or "unknown"
            )
            doc.metadata["doc_uuid"] = uuid
        grouped[uuid].append(doc)
    return grouped


def _serialize_document(doc: Document) -> dict:
    """Convert a Document into a JSON-serialisable dict."""
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
    }


def save_chunks_to_jsonl(
    docs: Sequence[Document], output_dir: Path | None = None
) -> None:
    """Save *docs* to disk as one *JSON Lines* file per original document."""
    if output_dir is None:
        output_dir = Path(CHUNKS_RAW_COLLECTION_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove any existing *.jsonl files so that chunks are always rewritten
    for existing in output_dir.glob("*.jsonl"):
        try:
            existing.unlink()
        except Exception as exc:
            print(f"! WARNING: could not delete {existing}: {exc}")

    grouped = _group_chunks_by_source(docs)

    for uuid_key, doc_list in grouped.items():
        # Save one JSONL per original document UUID
        file_name = f"{uuid_key}.jsonl"
        output_path = output_dir / file_name
        with open(output_path, "w", encoding="utf-8") as fh:
            for d in doc_list:
                json.dump(_serialize_document(d), fh, ensure_ascii=False)
                fh.write("\n")
        print(
            f"→ Saved {len(doc_list):>4} chunks to {output_path.relative_to(output_dir.parent.parent)}"
        )


# ------------------------------ Final summary -------------------------------


def _print_report() -> None:
    """Print a detailed summary of the chunking process."""
    if not _FILE_CHUNK_INFO:
        print("No files were processed, nothing to report.")
        return

    total_files = len(_FILE_CHUNK_INFO)
    total_chunks = sum(n for _, n in _FILE_CHUNK_INFO)

    print("\nChunking completed.\n")
    print("Summary of operations:")
    print(f"  • Files processed:             {total_files}")
    print(f"  • Chunks generated:            {total_chunks}")

    # Detailed breakdown ----------------------------------------------------
    name_w = max(10, max(len(name) for name, _ in _FILE_CHUNK_INFO))
    header = f"{'File name':<{name_w}}  {'Chunks':>6}"
    print("\nPer-file details:")
    print(header)
    print("-" * len(header))
    for name, count in sorted(_FILE_CHUNK_INFO):
        print(f"{name:<{name_w}}  {count:6d}")


# Persist the current run
save_chunks_to_jsonl(chunks)
_print_report()


# if __name__ == "__main__":
#     for chunk in chunks:
#         print(chunk.metadata)
#         print(chunk.page_content)
#         print("-" * 100)

# # %%
