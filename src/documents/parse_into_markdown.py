"""Script to parse PDFs to markdown and clean them with retries."""

# %%
import os
import random
import sys
import time
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
            print(
                f"Error en intento {attempt}/{max_retries}: {exc}. "
                "Reintentando si corresponde..."
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
            print(
                f"Error en intento {attempt}/{max_retries}: {exc}. "
                "Reintentando si corresponde..."
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
        return
    except (ClientError, BotoCoreError, Exception) as first_exc:
        print(
            "Fallo download_file; intentando fallback con get_object. "
            f"Error: {first_exc}"
        )

    # Fallback: usar get_object y escribir a disco manualmente
    def _get_object():
        return s3_client.get_object(Bucket=bucket_name, Key=object_key)

    response = call_with_retries(_get_object)
    body = response.get("Body")
    if body is None:
        raise RuntimeError("Respuesta S3 sin Body en get_object")

    with destination_path.open("wb") as file_obj:
        for chunk in iter(lambda: body.read(8 * 1024), b""):
            if not chunk:
                break
            file_obj.write(chunk)


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

    # Procesar los primeros 20 documentos
    for doc_idx, doc in enumerate(docs[97:]):
        print(f"\n========== Procesando documento {doc_idx + 1}/{len(docs)} ========")
        s3_key = doc.metadata["s3_key"]

        # Transformar el S3 key en ruta y extraer el nombre de archivo
        filename = Path(s3_key).name
        md_filename = Path(filename).with_suffix(".md")
        md_path = MARKDOWN_RAW_COLLECTION_DIR / md_filename

        # Si el markdown ya existe, cargarlo directamente y saltar la extracción
        if md_path.exists():
            print(f"Markdown {md_path} ya existe. Cargándolo desde disco.")
            with md_path.open(encoding="utf-8") as md_file:
                markdown_content = md_file.read()
            print(f"Markdown de {md_path} cargado. Longitud: {len(markdown_content)}")
        else:
            # Descarga el PDF únicamente si no existe localmente
            PDF_COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
            local_path = PDF_COLLECTION_DIR / filename
            if not local_path.exists():
                try:
                    download_s3_to_path_with_fallback(
                        s3, BUCKET_NAME, s3_key, local_path
                    )
                    print(f"Archivo guardado en {local_path}")
                except Exception as exc:
                    print(
                        f"Fallo persistente descargando desde S3 (bucket={BUCKET_NAME}, key={s3_key}). Error: {exc}"
                    )
                    # Omitir documento si no podemos descargar
                    continue
            else:
                print(f"PDF {local_path} ya existe. Usando copia local.")

            # Extraer markdown a partir del PDF usando Azure AI Document Intelligence
            load_dotenv(override=True)
            loader = AzureAIDocumentIntelligenceLoader(
                api_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                file_path=str(local_path),
                api_model="prebuilt-layout",  #
                mode="markdown",  #
                analysis_features=["ocrHighResolution"],  #
            )
            try:
                raw_doc = call_with_retries(loader.load)
            except Exception as first_exc:
                print(
                    "Fallo extrayendo markdown con prebuilt-layout; "
                    "probando fallback prebuilt-read/text. "
                    f"Error: {first_exc}"
                )
                # Fallback de extracción: usar prebuilt-read en modo texto
                read_loader = AzureAIDocumentIntelligenceLoader(
                    api_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    file_path=str(local_path),
                    api_model="prebuilt-read",
                    mode="text",
                )
                try:
                    raw_doc = call_with_retries(read_loader.load)
                except Exception as second_exc:
                    print(
                        "Fallo también con prebuilt-read/text. Omitiendo documento. "
                        f"Error: {second_exc}"
                    )
                    continue

            # Guardar el markdown extraído en disco para usos futuros
            MARKDOWN_RAW_COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
            with md_path.open("w", encoding="utf-8") as md_file:
                md_file.write(raw_doc[0].page_content)
            print(f"Markdown guardado en {md_path}")

            markdown_content = raw_doc[0].page_content

        # Limpiar el markdown con los modelos especificados
        clean_md = {}
        document_failed = False
        for model_name in MODELS_TO_CLEAN:
            print(f"Cleaning markdown with {model_name}...")
            chain_for_cleaning_markdown = prompt_for_cleaning_markdown | get_llm(
                model=model_name
            ).with_structured_output(CleanMarkdown)

            try:
                response = invoke_with_retries(
                    chain_for_cleaning_markdown,
                    {"markdown_content": markdown_content},
                )
                print(f"Finished cleaning markdown with {model_name}.")
                clean_md[model_name] = response.cleaned_markdown
            except Exception as exc:
                print(
                    "Fallo persistente limpiando markdown con "
                    f"{model_name} tras {MAX_RETRIES} intentos. "
                    f"Se omite el documento. Error: {exc}"
                )
                document_failed = True
                break

        if document_failed:
            # No guardar resultados parciales; continuar con el siguiente documento
            continue

        # Guardar los markdown limpios en la carpeta refined con sufijo del modelo
        MARKDOWN_REFINED_COLLECTION_DIR.mkdir(parents=True, exist_ok=True)
        for model_name, markdown_text in clean_md.items():
            refined_filename = f"{md_filename.stem}_{model_name}.md"
            refined_path = MARKDOWN_REFINED_COLLECTION_DIR / refined_filename
            with refined_path.open("w", encoding="utf-8") as refined_file:
                refined_file.write(markdown_text)
            print(f"Markdown limpio guardado en {refined_path}")


if __name__ == "__main__":
    process_documents()

# %%
