"""Utilities for counting PDF pages in the CSW-NVIRO document collection.

This script walks over every PDF inside ``src/documents/collections/pdf`` and
prints the number of PDFs found, the number of pages per document, and the
overall grand total of pages.

Run it from the project root with:

uv run -m src.documents.count_pdf_pages

(Using ``uv`` ensures the local virtual-environment is active, as mandated by
project conventions.)
"""

from __future__ import annotations

from pathlib import Path


try:
    from pypdf import PdfReader  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover
    message = "The 'pypdf' package is required. Install it with `uv pip install pypdf`"
    raise SystemExit(message) from exc


PDF_DIR = Path(__file__).resolve().parent / "collections" / "pdf"


def pages_in_pdf(pdf_path: Path) -> int:
    """Return the number of pages in *pdf_path*.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        The number of pages contained in the PDF.
    """
    with pdf_path.open("rb") as fp:
        reader = PdfReader(fp)
        return len(reader.pages)


def count_pages(directory: Path = PDF_DIR) -> None:
    """Print a page count report for every PDF in *directory*.

    Args:
        directory: Directory that holds PDF files. Defaults to
            ``src/documents/collections/pdf``.
    """
    if not directory.is_dir():
        raise SystemExit(f"Directory not found: {directory}")

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        raise SystemExit(f"No PDF files found under {directory}")

    pdf_count = len(pdf_files)
    print(f"Found {pdf_count} PDFs under {directory}\n")

    grand_total = 0
    print("Pages per PDF:\n--------------")

    for pdf_path in pdf_files:
        pages = pages_in_pdf(pdf_path)
        grand_total += pages
        print(f"{pdf_path.name:60s} {pages:6d} pages")

    print("--------------")
    print(f"Total across {pdf_count} PDFs: {grand_total} pages")


if __name__ == "__main__":  # pragma: no cover
    count_pages()
