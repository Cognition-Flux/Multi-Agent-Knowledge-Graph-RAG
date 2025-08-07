# %%
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    AzureAIDocumentIntelligenceLoader,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

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

# %%
# Prompt y modelo para limpiar markdown
PROMPT_TO_CLEAN_MARKDOWN = """
You are a helpful assistant that refines and improves the quality and structure of the markdown files.
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

# Procesar los primeros 20 documentos
for doc_idx, doc in enumerate(docs[20:]):
    print(f"\n========== Procesando documento {doc_idx + 1}/{len(docs)} ========")
    OBJECT_KEY = doc.metadata["s3_key"]

    # Transformar el S3 key en ruta y extraer el nombre de archivo
    filename = Path(OBJECT_KEY).name
    md_filename = Path(filename).with_suffix(".md")
    md_path = MARKDOWN_RAW_COLLECTION_DIR / md_filename

    # Si el markdown ya existe, cargarlo directamente y saltar la extracción
    if md_path.exists():
        print(f"Markdown {md_path} ya existe. Cargándolo desde disco.")
        with open(md_path, encoding="utf-8") as md_file:
            markdown_content = md_file.read()
        print(f"Markdown de {md_path} cargado. Longitud: {len(markdown_content)}")
    else:
        # Descarga el PDF únicamente si no existe localmente
        os.makedirs(PDF_COLLECTION_DIR, exist_ok=True)
        LOCAL_PATH = str(PDF_COLLECTION_DIR / filename)
        if not Path(LOCAL_PATH).exists():
            s3.download_file(BUCKET_NAME, OBJECT_KEY, LOCAL_PATH)
            print(f"Archivo guardado en {LOCAL_PATH}")
        else:
            print(f"PDF {LOCAL_PATH} ya existe. Usando copia local.")

        # Extraer markdown a partir del PDF usando Azure AI Document Intelligence
        load_dotenv(override=True)
        loader = AzureAIDocumentIntelligenceLoader(
            api_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            file_path=LOCAL_PATH,
            api_model="prebuilt-layout",  #
            mode="markdown",  #
            analysis_features=["ocrHighResolution"],  #
        )
        raw_doc = loader.load()

        # Guardar el markdown extraído en disco para usos futuros
        os.makedirs(MARKDOWN_RAW_COLLECTION_DIR, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as md_file:
            md_file.write(raw_doc[0].page_content)
        print(f"Markdown guardado en {md_path}")

        markdown_content = raw_doc[0].page_content

    # Limpiar el markdown con los modelos especificados
    clean_md = {}
    for m in MODELS_TO_CLEAN:
        print(f"Cleaning markdown with {m}...")
        chain_for_cleaning_markdown = prompt_for_cleaning_markdown | get_llm(
            model=m
        ).with_structured_output(CleanMarkdown)
        response = chain_for_cleaning_markdown.invoke(
            {"markdown_content": markdown_content}
        )
        print(f"Finished cleaning markdown with {m}.")
        clean_md[m] = response.cleaned_markdown

    # Guardar los markdown limpios en la carpeta refined con sufijo del modelo
    os.makedirs(MARKDOWN_REFINED_COLLECTION_DIR, exist_ok=True)
    for model_name, markdown_text in clean_md.items():
        refined_filename = f"{md_filename.stem}_{model_name}.md"
        refined_path = MARKDOWN_REFINED_COLLECTION_DIR / refined_filename
        with open(refined_path, "w", encoding="utf-8") as refined_file:
            refined_file.write(markdown_text)
        print(f"Markdown limpio guardado en {refined_path}")

# %%
