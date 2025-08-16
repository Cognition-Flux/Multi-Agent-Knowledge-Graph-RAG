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
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from src.config import CHUNKS_RAW_COLLECTION_DIR, CHUNKS_REFINED_COLLECTION_DIR
from src.documents.metadata import load_metadata


# -----------------------------------------------------------------------------
# Ingestion naming conventions (added for round_2 compatibility)
# -----------------------------------------------------------------------------
# Match the prefix 'round_2_<8-hex-uuid>_' at the start of filenames
_ROUND2_PREFIX_RE: re.Pattern = re.compile(r"^round_2_[0-9a-f]{8}_", re.IGNORECASE)
# Match filenames that consist solely of the 8-char UUID (e.g. '8fae1c34.jsonl')
_UUID_JSONL_RE: re.Pattern = re.compile(r"^[0-9a-f]{8}\.jsonl$", re.IGNORECASE)
# Match trailing model designators like '_gpt-4.1' (without extension)
_MODEL_SUFFIX_RE: re.Pattern = re.compile(r"_gpt-[0-9.]+$", re.IGNORECASE)


def _strip_round2_prefix(name: str) -> str:
    """Return *name* without the 'round_2_<uuid>_' prefix (if present)."""
    return _ROUND2_PREFIX_RE.sub("", name, count=1)


def _strip_model_suffix(name: str) -> str:
    """Return *name* without suffixes like '_gpt-4.1' (if present)."""
    return _MODEL_SUFFIX_RE.sub("", name, count=1)


# Default path for flora & fauna metadata Parquet file (imported from config)

###############################################################################
# Helpers                                                                     #
###############################################################################


def _collect_jsonl_files() -> list[Path]:
    """Return the list of *JSONL* chunk files for the current ingestion round.

    We first look for files whose **basename** starts with the
    ``round_2_<uuid>_`` prefix to keep datasets from different runs isolated.  If
    none are found, we gracefully fall back to *every* ``*.jsonl`` file to
    remain backward-compatible with older naming schemes.
    """
    directory = Path(CHUNKS_RAW_COLLECTION_DIR)
    if not directory.exists():
        raise FileNotFoundError(
            f"Directory {directory} does not exist. "
            "Make sure you have executed step01 to generate JSONL chunks."
        )

    # First, prefer the new naming scheme: one JSONL per UUID (8-hex chars)
    files = sorted(
        p
        for p in directory.glob("*.jsonl")
        if p.is_file() and _UUID_JSONL_RE.match(p.name)
    )

    if not files:
        # Backward-compat: older scheme with round_2_ prefix
        files = sorted(
            p
            for p in directory.glob("*.jsonl")
            if p.is_file() and _ROUND2_PREFIX_RE.match(p.name)
        )

    if not files:
        # Last resort – take everything
        print(
            "! WARNING: No JSONL files matched UUID or round_2_ patterns; "
            "processing every *.jsonl file in the directory."
        )
        files = sorted(p for p in directory.glob("*.jsonl") if p.is_file())

    return files


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
# chunk_dict = load_chunks_grouped()


# metadata_df = load_metadata()


def _simplify(s: str) -> str:
    """Return a simplified ASCII-only lowercase string with no separators."""
    # Remove ingestion-specific decorations before simplification
    s = _strip_round2_prefix(s)
    s = _strip_model_suffix(s)

    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\.pdf$", "", s)  # drop extension
    # Replace common separators with a single space then strip
    s = re.sub(r"[_\-\.]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Keep only alphanumerics
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


# ---------------------------------------------------------------------------
# Regex-based helpers for fuzzy filename matching
# ---------------------------------------------------------------------------


def _normalize_tokens(s: str) -> list[str]:
    """Return a list of alphanumeric tokens extracted from *s* (accent-stripped)."""
    # Strip prefix/suffix that are irrelevant for matching
    s = _strip_round2_prefix(s)
    s = _strip_model_suffix(s)

    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\.pdf$", "", s)  # drop extension
    # Split on common separators and digits boundaries
    tokens = re.split(r"[_\-\.\s]+", s)
    # Remove empty strings and trivial tokens
    return [t for t in tokens if t]


def _build_regex_from_tokens(tokens: list[str]) -> re.Pattern:
    """Return a compiled regex matching *tokens* in order allowing gaps."""
    if not tokens:
        return re.compile(r"^")  # will never match
    pattern = ".*".join(re.escape(t) for t in tokens)
    return re.compile(pattern, flags=re.IGNORECASE)


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


def _find_best_row(key: str, df: pd.DataFrame) -> tuple[pd.Series | None, str]:
    """Return the first matching metadata **row** for *key* using a multi-step strategy.

    1. Normalised containment check (fast).
    2. Regex match that tolerates different separators and extra text.
    3. Fallback to SequenceMatcher similarity if everything else fails.
    """
    key_tokens = _normalize_tokens(key)
    key_regex = _build_regex_from_tokens(key_tokens)

    best_match: pd.Series | None = None
    best_ratio = 0.0
    match_method = ""

    for _, row in df.iterrows():
        file_name_raw = str(row["file_name"])
        row_tokens = _normalize_tokens(file_name_raw)

        # 1. Direct token containment (quick win)
        if all(t in row_tokens for t in key_tokens) or all(
            t in key_tokens for t in row_tokens
        ):
            return row, "token_containment"

        # 2. Regex search (allows gaps / extra chars)
        if key_regex.search(" ".join(row_tokens)):
            return row, "regex_key"
        row_regex = _build_regex_from_tokens(row_tokens)
        if row_regex.search(" ".join(key_tokens)):
            return row, "regex_row"

        # 3. Similarity ratio as last resort
        from difflib import SequenceMatcher

        ratio = SequenceMatcher(None, "".join(key_tokens), "".join(row_tokens)).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = row if ratio > 0.8 else best_match  # threshold

    if best_match is not None:
        return best_match, "similarity"
    return None, "no_match"


def main() -> None:
    """Execute the main logic of the script."""
    # Immediately load grouped chunks when module is imported
    chunk_dict = load_chunks_grouped()
    metadata_df = load_metadata()

    # Enrich each Document with metadata from the DataFrame ----------------------
    # (file_key, n_docs, enriched?, matched_name, fields_updated, match_method)
    _file_info: list[tuple[str, int, bool, str | None, list[str], str]] = []
    _unmatched_files: list[str] = []
    _enriched_docs: int = 0

    def _derive_reference_name(docs: list[Document]) -> str:
        """Return a filename-like string to match against metadata.

        Uses the first document's *source_path* metadata to reconstruct the
        original PDF/markdown name (without the round_2 prefix or model suffix).
        Fallbacks to the JSONL key (UUID) if unavailable.
        """
        if not docs:
            return ""

        source_path = docs[0].metadata.get("source_path", "")
        basename = Path(source_path).name
        # Remove ingestion prefix and model suffix
        basename = _strip_round2_prefix(basename)
        basename = _strip_model_suffix(basename)
        return basename

    for jsonl_uuid, doc_list in chunk_dict.items():
        reference_name = _derive_reference_name(doc_list) or jsonl_uuid

        row, match_method = _find_best_row(reference_name, metadata_df)
        enriched = row is not None

        if not enriched:
            print(
                f"! WARNING: No metadata row matched for '{reference_name}' (simplified='{_simplify(reference_name)}')"
            )
            _unmatched_files.append(reference_name)
            _file_info.append(
                (jsonl_uuid, len(doc_list), False, None, [], match_method)
            )
            continue

        # When we reach here, we have a matching metadata row ------------------
        row_meta: dict = row.to_dict()
        fields_updated = list(row_meta.keys())
        matched_name = str(row_meta.get("file_name", "?"))

        # Attach each column as extra metadata keys
        for d in doc_list:
            d.metadata.update(row_meta)
        _enriched_docs += len(doc_list)
        _file_info.append(
            (
                jsonl_uuid,
                len(doc_list),
                True,
                matched_name,
                fields_updated,
                match_method,
            )
        )

    # ----------------------------- Persist augmented chunks ------------------
    save_chunks_grouped(chunk_dict)

    # ----------------------------- Final summary ------------------------------
    _total_docs = sum(len(v) for v in chunk_dict.values())
    _total_files = len(chunk_dict)

    print("\nMetadata enrichment completed.\n")
    print("Summary of operations:")
    print(f"  • Files processed:             {_total_files}")
    print(f"  • Documents restored:          {_total_docs}")
    _coverage_pct = (_enriched_docs / _total_docs * 100) if _total_docs else 0.0
    print(f"  • Documents enriched:          {_enriched_docs} ({_coverage_pct:.1f}% )")
    print(f"  • Files without metadata:      {len(_unmatched_files)}")

    # Detailed breakdown ------------------------------------------------------
    print("\nPer-file details:")

    if not _file_info:
        print("No files were processed.")
        return

    for key, n_docs, enriched, matched_name, fields, method in sorted(_file_info):
        print("-" * 80)
        print(f"  Chunk key          : {key}")
        if enriched:
            print("  Enriched           : Yes")
            print(f"  Matching file_name : {matched_name}")
            print(f"  Docs count         : {n_docs}")
            print(f"  Match method       : {method}")
            if fields:
                print("  Updated fields     :")
                # Print fields in 2 columns for better readability
                col1, col2 = [], []
                # Ensure fields are sorted for consistent output
                for i, f in enumerate(sorted(fields)):
                    if i % 2 == 0:
                        col1.append(f)
                    else:
                        col2.append(f)

                # Make columns of equal length for zipping
                if len(col1) > len(col2):
                    col2.append("")

                max_w1 = max(len(f) for f in col1) if col1 else 0
                for f1, f2 in zip(col1, col2, strict=False):
                    if f2:
                        print(f"    - {f1:<{max_w1}}   - {f2}")
                    else:
                        print(f"    - {f1}")
        else:
            print("  Enriched           : No")
            print(f"  Docs count         : {n_docs}")
            print(f"  Match method       : {method} (no match found)")
    print("-" * 80)

    if _unmatched_files:
        print("\nFiles without metadata (" + str(len(_unmatched_files)) + "):")
        for f in sorted(_unmatched_files):
            print(f"  - {f}")

    # Final Aggregated Summary ------------------------------------------------
    unique_source_docs = {info[3] for info in _file_info if info[2] and info[3]}
    num_unique_docs = len(unique_source_docs)

    print("\n\n" + "=" * 80)
    print("Final Aggregated Summary".center(80))
    print("-" * 80)
    print(f"  {'Total processed chunks':<45} | {_total_docs:>10}")
    print(f"  {'Unique source documents (PDFs)':<45} | {num_unique_docs:>10}")
    print("=" * 80)


if __name__ == "__main__":
    main()

# %%
