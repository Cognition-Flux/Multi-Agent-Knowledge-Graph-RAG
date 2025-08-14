# %%
import argparse
import pathlib


try:
    import fitz  # PyMuPDF preferred import name
except ImportError:  # compatibility fallback
    import pymupdf as fitz  # type: ignore

from markdownify import markdownify as md


# %%
def convert_pdf_to_markdown(pdf_path: pathlib.Path, output_dir: pathlib.Path):
    """Converts a PDF file to a Markdown file, stripping out image tags."""
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output markdown file path
    markdown_path = output_dir / f"{pdf_path.stem}.md"

    print(f"Opening PDF: {pdf_path}")

    # --------------------------
    # Step 1: Extract page text
    # --------------------------
    try:
        with fitz.open(str(pdf_path)) as doc:
            # Decide extraction mode – prefer direct Markdown if supported
            extraction_mode = "html"  # default
            try:
                _ = doc[0].get_text("markdown")
                extraction_mode = "markdown"
            except Exception:
                # Any issue -> fallback to HTML
                extraction_mode = "html"

            print(f"Using extraction mode: {extraction_mode}")

            content = ""
            for page in doc:
                if extraction_mode == "markdown":
                    content += page.get_text("markdown")
                else:
                    content += page.get_text("html")
    except Exception as e:
        print(f"Error processing PDF {pdf_path.name}: {e}")
        return

    # --------------------------
    # Step 2: Convert to Markdown
    # --------------------------
    if extraction_mode == "markdown":
        markdown_text = content

        # Strip inline image markdown e.g. ![alt](src)
        import re as _re

        markdown_text = _re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", markdown_text)
    else:
        html_text = content
        print("Converting HTML to Markdown (stripping images & non-content tags)...")

        # Prefer lxml parser if available for better HTML handling
        bs4_features = "lxml"
        try:
            import lxml  # noqa: F401
        except ModuleNotFoundError:
            bs4_features = "html.parser"

        markdown_text = md(
            html_text,
            strip=["img", "style", "head", "meta", "title", "script"],
            bs4_options={"features": bs4_features},
            table_infer_header=True,
        )

    import re

    # Remove repeated page headers/footers: lines occurring > 5 times
    lines = markdown_text.splitlines()
    freq = {}
    for line in lines:
        key = line.strip()
        if key:
            freq[key] = freq.get(key, 0) + 1

    threshold = 5
    cleaned_lines = [line for line in lines if freq.get(line.strip(), 0) <= threshold]
    markdown_text = "\n".join(cleaned_lines)

    # Collapse 3+ consecutive blank lines into at most 2
    markdown_text = re.sub(r"[\n\r]{3,}", "\n\n", markdown_text)

    # Write the final Markdown file
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    print(f"Successfully converted and saved to {markdown_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a PDF file to a Markdown file, stripping images."
    )
    parser.add_argument(
        "pdf_file",
        type=str,
        nargs="?",
        default="src/documents/collections/pdf/00.-Informe-T_e_cnico-Flora-y-Vegetaci_o_n-Valle-Noble.pdf",
        help="The path to the input PDF file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="localPDFparse/markdown",
        help="The directory to save the output Markdown file.",
    )

    args = parser.parse_args()

    pdf_path = pathlib.Path(args.pdf_file)
    output_dir = pathlib.Path(args.output_dir)

    if not pdf_path.is_file():
        print(f"Error: The file '{pdf_path}' does not exist.")
    else:
        convert_pdf_to_markdown(pdf_path, output_dir)
