# %%
"""Restore previously saved Markdown chunks from JSON Lines files.

This module complements `markdown_chunking_step01.py`.  While *step 01* takes
care of **splitting** Markdown into `langchain_core.documents.Document` objects
and persisting them as one **JSONL** file per original document, *step 02*
performs the inverse operation: it **loads** those JSONL files and rebuilds the
`Document` instances so they are ready for downstream use (vector stores,
search pipelines, etc.).

Usage (CLI):

```bash
uv run -m src.documents.markdown_chunking_step02          # prints summary
```

Programmatic API:

```python
from src.documents.markdown_chunking_step02 import load_all_chunks

chunks = load_all_chunks()  # list[Document]
```
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from src.config import CHUNKS_RAW_COLLECTION_DIR, CHUNKS_REFINED_COLLECTION_DIR
from src.documents.metadata import load_metadata


# Default path for flora & fauna metadata Parquet file (imported from config)

###############################################################################
# Helpers                                                                     #
###############################################################################


def _collect_jsonl_files() -> list[Path]:
    """Return a sorted list of all *.jsonl* files with raw chunks."""
    directory = Path(CHUNKS_RAW_COLLECTION_DIR)
    if not directory.exists():
        raise FileNotFoundError(
            f"Directory {directory} does not exist. "
            "Make sure you have executed step01 to generate JSONL chunks."
        )

    return sorted(p for p in directory.glob("*.jsonl") if p.is_file())


###############################################################################
# Deserialisation logic                                                        #
###############################################################################


def _deserialize_document(obj: dict) -> Document:
    """Convert a plain dict into a `Document`."""
    if "page_content" not in obj or "metadata" not in obj:
        raise ValueError(
            "JSON object must contain 'page_content' and 'metadata' keys to be "
            "converted into a Document."
        )

    return Document(page_content=obj["page_content"], metadata=obj["metadata"])


def load_chunks_from_file(path: str | Path) -> list[Document]:
    """Load every chunk from a single *JSONL* file located at *path*."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    documents: list[Document] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            obj = json.loads(line)
            documents.append(_deserialize_document(obj))
    return documents


def load_all_chunks() -> list[Document]:
    """Load *all* chunks across every JSONL file and return them concatenated."""
    all_docs: list[Document] = []
    for jsonl_path in _collect_jsonl_files():
        file_docs = load_chunks_from_file(jsonl_path)
        all_docs.extend(file_docs)
        print(f"✓ {jsonl_path.name:<40} ← {len(file_docs):>4} chunks restored")

    print("\nTotal chunks restored:", len(all_docs))
    return all_docs


def load_chunks_grouped() -> dict[str, list[Document]]:
    """Load chunks and return a dict[file_stem -> list[Document]]."""
    grouped: dict[str, list[Document]] = defaultdict(list)
    for jsonl_path in _collect_jsonl_files():
        file_docs = load_chunks_from_file(jsonl_path)
        key = jsonl_path.stem  # filename without extension
        grouped[key].extend(file_docs)
        print(f"✓ {key:<40} ← {len(file_docs):>4} chunks restored")

    total = sum(len(v) for v in grouped.values())
    print("\nTotal chunks restored:", total)
    return grouped


# Immediately load grouped chunks when module is imported
chunk_dict = load_chunks_grouped()


metadata_df = load_metadata()

import re
import textwrap
import unicodedata


def _simplify(s: str) -> str:
    """Return a simplified ASCII-only lowercase string without separators."""
    # Remove diacritics (accents) and convert to ASCII
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\.pdf$", "", s)  # drop extension
    # Replace common separators with a single space then strip
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Keep only alphanumerics
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def _serialize_document(doc: Document) -> dict:
    """Return a JSON-serialisable dict for a Document."""
    return {"page_content": doc.page_content, "metadata": doc.metadata}


def save_chunks_grouped(grouped: dict[str, list[Document]]) -> None:
    """Persist each group of enriched chunks as JSONL in *refined* directory."""
    CHUNKS_REFINED_COLLECTION_DIR.mkdir(parents=True, exist_ok=True)

    for key, docs in grouped.items():
        out_path = CHUNKS_REFINED_COLLECTION_DIR / f"{key}_augmented.jsonl"
        with open(out_path, "w", encoding="utf-8") as fh:
            for doc in docs:
                json.dump(_serialize_document(doc), fh, ensure_ascii=False)
                fh.write("\n")
        print(f"✓ {out_path.name:<40} → {len(docs):>4} chunks saved")


def _find_best_row(key: str, df: pd.DataFrame) -> pd.Series | None:
    """Return the first row whose file_name contains *key* (fuzzy)."""
    key_simple = _simplify(key)
    best_match: pd.Series | None = None
    for _, row in df.iterrows():
        row_simple = _simplify(str(row["file_name"]))
        # Direct containment in either direction
        if key_simple in row_simple or row_simple in key_simple:
            return row
        # Fallback: similarity ratio
        from difflib import SequenceMatcher

        ratio = SequenceMatcher(None, key_simple, row_simple).ratio()
        if ratio > 0.8:  # heuristic threshold
            best_match = row
            break
    return best_match


# Enrich each Document with metadata from the DataFrame ----------------------
# (file_key, n_docs, enriched?, matched_name, fields_updated)
_file_info: list[tuple[str, int, bool, str | None, list[str]]] = []
_unmatched_files: list[str] = []
_enriched_docs: int = 0

for doc_key, doc_list in chunk_dict.items():
    row = _find_best_row(doc_key, metadata_df)
    enriched = row is not None

    if not enriched:
        print(
            f"! WARNING: No metadata row matched for '{doc_key}' (simplified='{_simplify(doc_key)}')"
        )
        _unmatched_files.append(doc_key)
        _file_info.append((doc_key, len(doc_list), False, None, []))
        continue

    # When we reach here, we have a matching metadata row ------------------
    row_meta: dict = row.to_dict()
    fields_updated = list(row_meta.keys())
    matched_name = str(row_meta.get("file_name", "?"))

    # Attach each column as extra metadata keys
    for d in doc_list:
        d.metadata.update(row_meta)
    _enriched_docs += len(doc_list)
    _file_info.append((doc_key, len(doc_list), True, matched_name, fields_updated))

# ----------------------------- Persist augmented chunks ------------------
save_chunks_grouped(chunk_dict)

# ----------------------------- Final summary ------------------------------
_total_docs = sum(len(v) for v in chunk_dict.values())
_total_files = len(chunk_dict)

print("\nMetadata enrichment completed.\n")
print("Summary of operations:")
print(f"  • Files processed:             {_total_files}")
print(f"  • Documents restored:          {_total_docs}")
print(f"  • Documents enriched:          {_enriched_docs}")
print(f"  • Files without metadata:      {len(_unmatched_files)}")

# Detailed breakdown ------------------------------------------------------
print("\nPer-file details:")

# Dynamic column widths for tidy layout
_key_w = max(9, max(len(k) for k, *_ in _file_info))
_meta_w = max(17, max(len(m or "") for *_, m, _ in _file_info))

header = (
    f"{'Chunk key':<{_key_w}}  "
    f"{'Metadata file_name':<{_meta_w}}  "
    f"{'Docs':>5}  {'Enr.':>5}  Updated fields"
)
print(header)
print("-" * len(header))

for key, n_docs, enriched, matched_name, fields in sorted(_file_info):
    flag = "yes" if enriched else "no"
    meta_name_display = matched_name if enriched else "—"
    row_prefix = (
        f"{key:<{_key_w}}  {meta_name_display:<{_meta_w}}  {n_docs:5d}  {flag:>5}"
    )

    if fields:
        field_str = ", ".join(fields)
        wrapped_lines = textwrap.wrap(field_str, width=100)
        print(f"{row_prefix}  {wrapped_lines[0]}")
        for cont in wrapped_lines[1:]:
            print(" " * (len(row_prefix) + 2) + cont)
    else:
        print(row_prefix)

if _unmatched_files:
    print("\nFiles without metadata →", ", ".join(sorted(_unmatched_files)))
