"""Script to parse PDFs to markdown and clean them with retries.

Uso:
uv run src/documents/parse_into_markdown.py



"""

# %%
import os
import random
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# %%
# Prompt y modelo para limpiar markdown
PROMPT_TO_CLEAN_MARKDOWN = """
You are a helpful assistant that refines and improves the quality and
structure of the markdown files.
You have to achieve a fully human readable markdown text.
ALWAYS do the following:
    - keep the ENTIRE content and data of all tables, always preserve it.
    - keep the ENTIRE content and information of all texts, always preserve it.

PROHIBITIONS:
    - DO NOT remove any content from the markdown file, NEVER DO IT.
    - DO NOT remove any data from the markdown file, NEVER DO IT.
    - DO NOT remove any information from the markdown file, NEVER DO IT.
    - DO NOT remove any metadata from the markdown file, NEVER DO IT.
"""

prompt_for_cleaning_markdown = ChatPromptTemplate.from_messages(
    [
        ("system", PROMPT_TO_CLEAN_MARKDOWN),
        ("human", "{markdown_content}"),
    ]
)


class CleanMarkdown(BaseModel):
    """The cleaned markdown string."""

    cleaned_markdown: str = Field(description="The cleaned markdown string.")


MODELS_TO_CLEAN = ["gpt-4.1"]  # , "gpt-4.1-mini"]  # "o3-mini",

# Parámetros de reintentos
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 2.0


def log_event(event: str, **kwargs) -> None:
    """Imprime eventos con timestamp y contexto clave=valor."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if kwargs:
        extras = " ".join(f"{key}={value}" for key, value in kwargs.items())
        print(f"[{timestamp}] {event} {extras}")
    else:
        print(f"[{timestamp}] {event}")


def invoke_with_retries(
    runnable,
    input_payload,
    *,
    max_retries: int = MAX_RETRIES,
) -> CleanMarkdown:
    """Invoca un runnable con reintentos y backoff exponencial con jitter.

    Lanza la última excepción si se agotan los intentos.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return runnable.invoke(input_payload)
        except Exception as exc:
            last_exc = exc
            log_event(
                "retry.invoke.error",
                attempt=f"{attempt}/{max_retries}",
                error=str(exc),
            )
            if attempt < max_retries:
                sleep_seconds = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                sleep_seconds += random.uniform(0.0, 0.5)
                time.sleep(sleep_seconds)
    # Si llegamos aquí, agotamos los intentos
    assert last_exc is not None
    raise last_exc


def call_with_retries(action, *, max_retries: int = MAX_RETRIES):
    """Ejecuta una acción con reintentos y backoff exponencial con jitter."""
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return action()
        except Exception as exc:
            last_exc = exc
            log_event(
                "retry.action.error",
                attempt=f"{attempt}/{max_retries}",
                error=str(exc),
            )
            if attempt < max_retries:
                sleep_seconds = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                sleep_seconds += random.uniform(0.0, 0.5)
                time.sleep(sleep_seconds)
    assert last_exc is not None
    raise last_exc


def download_s3_to_path_with_fallback(
    s3_client,
    bucket_name: str,
    object_key: str,
    destination_path: Path,
) -> None:
    """Descarga un objeto S3 a disco con reintentos y fallback sin HEAD.

    Intenta primero download_file (requiere HeadObject). Si da 403/HeadObject
    u otro error, hace fallback a get_object y escribe por streaming.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    def _download_file():
        return s3_client.download_file(bucket_name, object_key, str(destination_path))

    try:
        call_with_retries(_download_file)
        log_event(
            "pdf.download.success",
            method="download_file",
            bucket=bucket_name,
            key=object_key,
            path=str(destination_path),
        )
        return
    except (ClientError, BotoCoreError, Exception) as first_exc:
        log_event(
            "pdf.download.fallback",
            reason="download_file_failed",
            error=str(first_exc),
            bucket=bucket_name,
            key=object_key,
            path=str(destination_path),
        )

    # Fallback: usar get_object y escribir a disco manualmente
    def _get_object():
        return s3_client.get_object(Bucket=bucket_name, Key=object_key)

    response = call_with_retries(_get_object)
    body = response.get("Body")
    if body is None:
        raise RuntimeError("Respuesta S3 sin Body en get_object")

    with destination_path.open("wb") as file_obj:
        log_event(
            "pdf.download.streaming.start",
            method="get_object",
            bucket=bucket_name,
            key=object_key,
            path=str(destination_path),
        )
        for chunk in iter(lambda: body.read(8 * 1024), b""):
            if not chunk:
                break
            file_obj.write(chunk)
    log_event(
        "pdf.download.streaming.success",
        method="get_object",
        bucket=bucket_name,
        key=object_key,
        path=str(destination_path),
    )


def process_documents() -> None:
    """Procesa documentos descargando PDFs, extrayendo y limpiando markdown."""
    # Asegurar que el repo root esté en sys.path para permitir imports `src.*`
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Importar dependencias del proyecto tras asegurar PYTHONPATH
    from src.config import (
        BUCKET_NAME,
        MARKDOWN_RAW_COLLECTION_DIR,
        MARKDOWN_REFINED_COLLECTION_DIR,
        PDF_COLLECTION_DIR,
    )
    from src.documents.metadata import load_metadata
    from src.utils import get_llm

    # Cargar variables de entorno
    load_dotenv(override=True)

    # Cargar metadatos y preparar documentos
    metadata = load_metadata()
    docs = [
        Document(metadata=project.to_dict(), page_content="")
        for _i, project in metadata.iterrows()
    ]

    # Sesión para acceder a S3
    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name="us-west-2",
    )
    s3 = session.client("s3")

    # Procesar documentos (permite offset para reanudar)
    start_from = 0
    docs_to_process = docs[start_from:]
    total_docs = len(docs)
    for doc_idx, doc in enumerate(docs_to_process):
        index_abs = start_from + doc_idx + 1
        log_event("doc.process.start", index=f"{index_abs}/{total_docs}")
        s3_key = doc.metadata["s3_key"]

        # Transformar el S3 key en ruta y derivar nombre único con prefijo round_2
        original_filename = Path(s3_key).name
        unique_id = uuid.uuid4().hex[:8]
        filename = Path(f"round_2_{unique_id}_{original_filename}")
        md_filename = filename.with_suffix(".md")
        md_path = MARKDOWN_RAW_COLLECTION_DIR / md_filename

        # Si todos los resultados refinados ya existen para los modelos objetivos,
        # saltar por completo este documento para ahorrar trabajo.
        refined_paths_by_model = {
            model_name: MARKDOWN_REFINED_COLLECTION_DIR
            / f"{md_filename.stem}_{model_name}.md"
            for model_name in MODELS_TO_CLEAN
        }
        if all(path.exists() for path in refined_paths_by_model.values()):
            log_event(
                "doc.skip.refined_exists",
                file=filename,
            )
            log_event("doc.process.end", index=f"{index_abs}/{total_docs}")
            continue

        # Determinar si necesitamos parsear (extraer) o podemos reutilizar el markdown existente
        markdown_content = None
        needs_parsing = True
        if md_path.exists():
            log_event("md.raw.exists", path=str(md_path))
            with md_path.open(encoding="utf-8") as md_file:
                markdown_content = md_file.read()
            log_event("md.raw.loaded", path=str(md_path), length=len(markdown_content))
            if markdown_content.strip():
                needs_parsing = False
            else:
                log_event("md.raw.invalid_empty", path=str(md_path))

        if needs_parsing:
            # Descarga el PDF únicamente si no existe localmente
            PDF_COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
            local_path = PDF_COLLECTION_DIR / filename
            if not local_path.exists():
                try:
                    download_s3_to_path_with_fallback(
                        s3, BUCKET_NAME, s3_key, local_path
                    )
                    log_event("pdf.saved", path=str(local_path))
                except Exception as exc:
                    log_event(
                        "pdf.download.error",
                        bucket=BUCKET_NAME,
                        key=s3_key,
                        error=str(exc),
                    )
                    log_event("doc.process.end", index=f"{index_abs}/{total_docs}")
                    # Omitir documento si no podemos descargar
                    continue
            else:
                log_event("pdf.exists", path=str(local_path))

            # Extraer markdown: modo SaaS (Azure) o local según config
            from src.config import (
                USE_SAAS_PDF_PARSER,  # importar dentro para evitar ciclos
            )

            if USE_SAAS_PDF_PARSER:
                load_dotenv(override=True)
                loader = AzureAIDocumentIntelligenceLoader(
                    api_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    file_path=str(local_path),
                    api_model="prebuilt-layout",
                    mode="markdown",
                    analysis_features=["ocrHighResolution"],
                )
            else:
                from localPDFparse.parse import LocalPDFMarkdownLoader

                loader = LocalPDFMarkdownLoader(file_path=str(local_path))

            log_event(
                "md.extract.start",
                model="prebuilt-layout" if USE_SAAS_PDF_PARSER else "local",
                path=str(local_path),
            )
            try:
                raw_doc = call_with_retries(loader.load)
                # Validaciones de robustez: si la extracción no devuelve contenido utilizable
                # continuar con el siguiente documento.
                if (
                    not raw_doc
                    or len(raw_doc) == 0
                    or not getattr(raw_doc[0], "page_content", "").strip()
                ):
                    log_event("md.extract.empty_or_invalid", path=str(local_path))
                    log_event("doc.process.end", index=f"{index_abs}/{total_docs}")
                    continue
            except Exception as first_exc:
                if not USE_SAAS_PDF_PARSER:
                    # En modo local no hay fallback; registrar y continuar
                    log_event(
                        "md.extract.error_local",
                        path=str(local_path),
                        error=str(first_exc),
                    )
                    log_event("doc.process.end", index=f"{index_abs}/{total_docs}")
                    continue

                log_event(
                    "md.extract.error_primary",
                    model="prebuilt-layout",
                    path=str(local_path),
                    error=str(first_exc),
                )
                # Fallback de extracción: usar prebuilt-read en modo texto
                if USE_SAAS_PDF_PARSER:
                    read_loader = AzureAIDocumentIntelligenceLoader(
                        api_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                        file_path=str(local_path),
                        api_model="prebuilt-read",
                        mode="text",
                    )
                else:
                    # Si no es SaaS, no intentamos fallback de Azure
                    log_event("doc.process.end", index=f"{index_abs}/{total_docs}")
                    continue

                log_event(
                    "md.extract.fallback.start",
                    model="prebuilt-read",
                    path=str(local_path),
                )
                try:
                    raw_doc = call_with_retries(read_loader.load)
                    if (
                        not raw_doc
                        or len(raw_doc) == 0
                        or not getattr(raw_doc[0], "page_content", "").strip()
                    ):
                        log_event(
                            "md.extract.empty_or_invalid_fallback",
                            model="prebuilt-read",
                            path=str(local_path),
                        )
                        log_event("doc.process.end", index=f"{index_abs}/{total_docs}")
                        continue
                except Exception as second_exc:
                    log_event(
                        "md.extract.error_fallback",
                        model="prebuilt-read",
                        path=str(local_path),
                        error=str(second_exc),
                    )
                    log_event("doc.process.end", index=f"{index_abs}/{total_docs}")
                    continue

            # Guardar el markdown extraído en disco para usos futuros
            MARKDOWN_RAW_COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
            with md_path.open("w", encoding="utf-8") as md_file:
                md_file.write(raw_doc[0].page_content)
            log_event("md.raw.saved", path=str(md_path))

            markdown_content = raw_doc[0].page_content

        # Limpiar el markdown solo para los modelos cuyo resultado no exista aún
        models_to_run = []
        for model_name in MODELS_TO_CLEAN:
            refined_filename = f"{md_filename.stem}_{model_name}.md"
            refined_path = MARKDOWN_REFINED_COLLECTION_DIR / refined_filename
            if refined_path.exists():
                log_event(
                    "md.refined.exists",
                    model=model_name,
                    path=str(refined_path),
                )
                continue
            models_to_run.append(model_name)

        if not models_to_run:
            log_event("md.clean.skip_all_exist")
            log_event("doc.process.end", index=f"{index_abs}/{total_docs}")
            continue

        # Asegurar directorio de salida para refinados
        MARKDOWN_REFINED_COLLECTION_DIR.mkdir(parents=True, exist_ok=True)

        for model_name in models_to_run:
            log_event("md.clean.start", model=model_name)
            chain_for_cleaning_markdown = prompt_for_cleaning_markdown | get_llm(
                model=model_name
            ).with_structured_output(CleanMarkdown)

            try:
                response = invoke_with_retries(
                    chain_for_cleaning_markdown,
                    {"markdown_content": markdown_content},
                )
                log_event("md.clean.success", model=model_name)
                refined_filename = f"{md_filename.stem}_{model_name}.md"
                refined_path = MARKDOWN_REFINED_COLLECTION_DIR / refined_filename
                cleaned_text = response.cleaned_markdown or ""
                if not cleaned_text.strip():
                    log_event(
                        "md.clean.empty_result",
                        model=model_name,
                        path=str(refined_path),
                    )
                    continue
                with refined_path.open("w", encoding="utf-8") as refined_file:
                    refined_file.write(cleaned_text)
                log_event("md.refined.saved", model=model_name, path=str(refined_path))
            except Exception as exc:
                log_event("md.clean.error", model=model_name, error=str(exc))
                continue

        # Fin del procesamiento de este documento
        log_event("doc.process.end", index=f"{index_abs}/{total_docs}")


if __name__ == "__main__":
    process_documents()

# %%
