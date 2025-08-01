# %%
import os
from dotenv import load_dotenv
import boto3
from pathlib import Path
from src.config import PDF_COLLECTION_DIR, MARKDOWN_RAW_COLLECTION_DIR, BUCKET_NAME

from langchain_core.documents import Document


from src.documents.metadata import load_metadata


from dotenv import load_dotenv

from langchain_community.document_loaders import (
    AzureAIDocumentIntelligenceLoader,
)

from src.utils import get_llm

load_dotenv(override=True)

metadata = load_metadata()
docs = []
for i, project in metadata.iterrows():
    docs.append(Document(metadata=project.to_dict(), page_content=""))

session = boto3.Session(
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name="us-west-2",
)
s3 = session.client("s3")
OBJECT_KEY = docs[0].metadata["s3_key"]

# Transform the S3 key into a path and extract the filename
filename = Path(OBJECT_KEY).name
LOCAL_PATH = str(PDF_COLLECTION_DIR / filename)

# Descarga
os.makedirs(PDF_COLLECTION_DIR, exist_ok=True)
s3.download_file(BUCKET_NAME, OBJECT_KEY, LOCAL_PATH)
print(f"Archivo guardado en {LOCAL_PATH}")

load_dotenv(override=True)
loader = AzureAIDocumentIntelligenceLoader(
    api_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    file_path=LOCAL_PATH,
    api_model="prebuilt-layout",  # tablas
    mode="markdown",  # markdown, page o single
    analysis_features=["ocrHighResolution"],  # mejor OCR para texto pequeño
    # pages=None,  # "1-3,5" para limitar páginas
    # extra_headers={"x-ms-useragent": "langchain-loader/opt"},
)
raw_doc = loader.load()

# Guardar markdown
os.makedirs(MARKDOWN_RAW_COLLECTION_DIR, exist_ok=True)
md_filename = Path(filename).with_suffix(".md")
md_path = MARKDOWN_RAW_COLLECTION_DIR / md_filename
with open(md_path, "w", encoding="utf-8") as md_file:
    md_file.write(raw_doc[0].page_content)
print(f"Markdown guardado en {md_path}")

with open(md_path, "r", encoding="utf-8") as md_file:
    markdown_content = md_file.read()
print(f"Markdown de {md_path} cargado. Longitud: {len(markdown_content)}")

# %%

llm = get_llm()
# %%
