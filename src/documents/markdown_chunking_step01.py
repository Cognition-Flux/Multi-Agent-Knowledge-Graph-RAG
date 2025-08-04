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

import os
from pathlib import Path
from typing import List, Sequence

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from src.config import MARKDOWN_REFINED_COLLECTION_DIR, CHUNKS_RAW_COLLECTION_DIR

# Initialise environment variables (for local debugging if needed)
load_dotenv(override=True)

###############################################################################
# Helpers                                                                     #
###############################################################################


def collect_markdown_files() -> List[Path]:
    """Return a list with paths to all refined Markdown files."""

    directory = Path(MARKDOWN_REFINED_COLLECTION_DIR)
    if not directory.exists():
        raise FileNotFoundError(
            f"Directory {directory} does not exist. "
            "Make sure the refined Markdown collection has been generated."
        )

    # Collect every file that ends with .md inside the directory (non-recursive)
    return sorted(p for p in directory.glob("*.md") if p.is_file())


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


def _chunk_single_markdown(text: str, source_path: Path) -> Sequence[Document]:
    """Split *text* into quality chunks preserving header metadata.

    Args:
        text:      Full Markdown string.
        source_path: Path to the original Markdown file (added to metadata).

    Returns:
        A sequence of `Document` objects with ``page_content`` ≤ ``chunk_size``
        and rich metadata (heading hierarchy, source, chunk index).
    """

    header_docs = _HEADER_SPLITTER.split_text(text)

    chunks: list[Document] = []
    for doc in header_docs:
        # Further split if this header section is still too large
        sub_texts = _CHAR_SPLITTER.split_text(doc.page_content)
        for idx, chunk_text in enumerate(sub_texts):
            metadata = {
                **doc.metadata,  # heading hierarchy (e.g., {"h1": ..., "h2": ...})
                "source_path": str(source_path),
                "chunk_index": idx,
            }
            chunks.append(Document(page_content=chunk_text, metadata=metadata))
    return chunks


def chunk_all_markdown_files() -> List[Document]:
    """Chunk every Markdown file inside *MARKDOWN_REFINED_COLLECTION_DIR*."""

    all_chunks: list[Document] = []
    for md_path in collect_markdown_files():
        with open(md_path, "r", encoding="utf-8") as fh:
            text = fh.read()
        file_chunks = _chunk_single_markdown(text, md_path)
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
    grouped: dict[str, list[Document]] = defaultdict(list)
    for doc in docs:
        source = doc.metadata.get("source_path", "unknown")
        grouped[source].append(doc)
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

    grouped = _group_chunks_by_source(docs)

    for source_path, doc_list in grouped.items():
        # Use the stem of the original file for naming
        file_name = Path(source_path).stem + ".jsonl"
        output_path = output_dir / file_name
        with open(output_path, "w", encoding="utf-8") as fh:
            for d in doc_list:
                json.dump(_serialize_document(d), fh, ensure_ascii=False)
                fh.write("\n")
        print(
            f"→ Saved {len(doc_list):>4} chunks to {output_path.relative_to(output_dir.parent.parent)}"
        )


# Persist the current run
save_chunks_to_jsonl(chunks)


if __name__ == "__main__":
    for chunk in chunks:
        print(chunk.metadata)
        print(chunk.page_content)
        print("-" * 100)

# %%
