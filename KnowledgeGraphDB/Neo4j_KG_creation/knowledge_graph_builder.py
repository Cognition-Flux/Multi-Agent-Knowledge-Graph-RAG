# %%
from __future__ import annotations

import contextlib
import os
import re
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.cohere import CohereEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.indexes import create_fulltext_index, create_vector_index
from neo4j_graphrag.llm import AzureOpenAILLM

from src.config import CHUNKS_REFINED_COLLECTION_DIR
from src.documents.markdown_chunking_step02 import load_chunks_from_file
from src.utils import get_llm


load_dotenv(override=True)

NEO4J_USERNAME: str | None = os.getenv("NEO4J_USERNAME_UPGRADED")
NEO4J_PASSWORD: str | None = os.getenv("NEO4J_PASSWORD_UPGRADED")
NEO4J_URI: str | None = os.getenv("NEO4J_CONNECTION_URI_UPGRADED")

if not (NEO4J_USERNAME and NEO4J_PASSWORD and NEO4J_URI):
    raise OSError("⚠️  Variables de entorno de Neo4j incompletas. Revisa `.env`.")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
# Verificamos conectividad antes de proseguir.
with driver as _tmp_driver:
    _tmp_driver.verify_connectivity()

try:
    # Preferimos la implementación de Graphrag que cumple con LLMInterface
    llm = AzureOpenAILLM(
        model_name="gpt-4.1-mini",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
except Exception:
    # Fallback a wrapped LangChain model (no recomendado, pero evita fallo en local)
    llm = get_llm()
embedder = CohereEmbeddings(model="embed-v4.0", api_key=os.getenv("COHERE_API_KEY"))
text_splitter = FixedSizeSplitter(chunk_size=1024, chunk_overlap=32)

# --------------------------------------------------------------------------- #
# 2.1) Crear índices vectoriales y full-text para los nodos Chunk
# --------------------------------------------------------------------------- #

VECTOR_INDEX_NAME = "chunkEmbedding"
FULLTEXT_INDEX_NAME = "chunkFulltext"

# Intentamos inferir la dimensión automáticamente.
try:
    VECTOR_DIM = len(embedder.embed_query("test"))
except Exception:
    VECTOR_DIM = 1024  # fallback razonable

# Crear índices si no existen (idempotente)
with contextlib.suppress(Exception):
    create_vector_index(
        driver,
        name=VECTOR_INDEX_NAME,
        label="Chunk",
        embedding_property="embedding",
        dimensions=VECTOR_DIM,
        similarity_fn="cosine",
        fail_if_exists=False,
    )

with contextlib.suppress(Exception):
    create_fulltext_index(
        driver,
        name=FULLTEXT_INDEX_NAME,
        label="Chunk",
        node_properties=["text"],
        fail_if_exists=False,
    )

# --------------------------------------------------------------------------- #
# 2.2) Restaurar chunks agrupados por documento
# --------------------------------------------------------------------------- #


def _collect_refined_jsonl_files() -> list[Path]:
    """Return sorted list of *.jsonl* files in the refined chunks directory."""
    directory = Path(CHUNKS_REFINED_COLLECTION_DIR)
    if not directory.exists():
        raise FileNotFoundError(
            f"Directorio {directory} no existe. Ejecuta el script step02 para generar chunks refinados."
        )
    return sorted(p for p in directory.glob("*.jsonl") if p.is_file())


def restore_chunks_grouped() -> dict[str, list[Document]]:
    """Load chunks from refined JSONL files and group them by originating document."""
    grouped: dict[str, list[Document]] = defaultdict(list)
    for jsonl_path in _collect_refined_jsonl_files():
        key = jsonl_path.stem
        if key.endswith("_augmented"):
            key = key[: -len("_augmented")]
        docs = load_chunks_from_file(jsonl_path)
        grouped[key].extend(docs)
    return grouped


# Carga chunks inmediatamente para su uso posterior
chunk_dict = restore_chunks_grouped()
print(
    f"✓ Chunks restaurados: {sum(len(v) for v in chunk_dict.values())} en {len(chunk_dict)} documentos."
)
# --------------------------------------------------------------------------- #
# 2.3) Definición del esquema del Knowledge Graph
# --------------------------------------------------------------------------- #
# %%
NODE_TYPES = [
    {
        "label": "Project",
        "description": "Documento o proyecto tramitado en el SEA",
        "properties": [
            {"name": "name", "type": "STRING", "required": True},
            {"name": "id", "type": "INTEGER", "required": True},
        ],
    },
    {
        "label": "Chunk",
        "description": "Fragmento de texto del documento",
        "properties": [
            {"name": "text", "type": "STRING", "required": True},
            {"name": "chunk_index", "type": "INTEGER"},
            {"name": "h1", "type": "STRING"},
            {"name": "source_path", "type": "STRING"},
        ],
    },
    {
        "label": "Region",
        "properties": [{"name": "name", "type": "STRING", "required": True}],
    },
    {
        "label": "Commune",
        "properties": [{"name": "name", "type": "STRING", "required": True}],
    },
    {
        "label": "Tipologia",
        "properties": [{"name": "code", "type": "STRING", "required": True}],
    },
    {
        "label": "ProjectType",
        "properties": [{"name": "name", "type": "STRING", "required": True}],
    },
    {
        "label": "PresentationDate",
        "properties": [{"name": "date", "type": "DATE", "required": True}],
    },
    {
        "label": "DocumentType",
        "properties": [{"name": "name", "type": "STRING", "required": True}],
    },
    {
        "label": "DocumentSubtype",
        "properties": [{"name": "name", "type": "STRING", "required": True}],
    },
]

RELATIONSHIP_TYPES = [
    "HAS_CHUNK",
    "IN_REGION",
    "IN_COMMUNE",
    "HAS_TIPOLOGIA",
    "HAS_PROJECT_TYPE",
    "PRESENTED_ON",
    "HAS_DOCUMENT_TYPE",
    "HAS_DOCUMENT_SUBTYPE",
]

PATTERNS = [
    ("Project", "HAS_CHUNK", "Chunk"),
    ("Project", "IN_REGION", "Region"),
    ("Project", "IN_COMMUNE", "Commune"),
    ("Project", "HAS_TIPOLOGIA", "Tipologia"),
    ("Project", "HAS_PROJECT_TYPE", "ProjectType"),
    ("Project", "PRESENTED_ON", "PresentationDate"),
    ("Project", "HAS_DOCUMENT_TYPE", "DocumentType"),
    ("Project", "HAS_DOCUMENT_SUBTYPE", "DocumentSubtype"),
]


def ensure_property_indexes(_driver: GraphDatabase.driver) -> None:
    """Crea índices de propiedades para consultas y filtros eficientes."""
    index_statements = [
        "CREATE INDEX project_id_idx IF NOT EXISTS FOR (p:Project) ON (p.id)",
        "CREATE INDEX project_name_idx IF NOT EXISTS FOR (p:Project) ON (p.name)",
        "CREATE INDEX region_name_idx IF NOT EXISTS FOR (r:Region) ON (r.name)",
        "CREATE INDEX commune_name_idx IF NOT EXISTS FOR (c:Commune) ON (c.name)",
        "CREATE INDEX tipologia_code_idx IF NOT EXISTS FOR (t:Tipologia) ON (t.code)",
        "CREATE INDEX projecttype_name_idx IF NOT EXISTS FOR (pt:ProjectType) ON (pt.name)",
        "CREATE INDEX doctype_name_idx IF NOT EXISTS FOR (dt:DocumentType) ON (dt.name)",
        "CREATE INDEX docsubtype_name_idx IF NOT EXISTS FOR (ds:DocumentSubtype) ON (ds.name)",
        "CREATE INDEX presentation_date_idx IF NOT EXISTS FOR (d:PresentationDate) ON (d.date)",
    ]
    with _driver.session() as session:
        for stmt in index_statements:
            session.run(stmt)


def _parse_communes(communes_value: str | list[str] | None) -> list[str]:
    """Normaliza y divide el campo de comunas en una lista.

    Soporta separadores (hyphen/dash), slash y coma, y combina múltiples separadores.
    Elimina espacios sobrantes y entradas vacías.
    """
    if communes_value is None:
        return []
    if isinstance(communes_value, list):
        return [c.strip() for c in communes_value if isinstance(c, str) and c.strip()]
    s = str(communes_value)
    parts = re.split(r"[,/\-–—]+", s)
    return [p.strip() for p in parts if p.strip()]


def _iso_date_to_map(date_str: str | None) -> dict[str, int] | None:
    """Convierte 'YYYY-MM-DD' en un mapa {year, month, day} para Cypher date()."""
    if not date_str or not isinstance(date_str, str):
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", date_str)
    if not m:
        return None
    year, month, day = m.groups()
    try:
        return {"year": int(year), "month": int(month), "day": int(day)}
    except Exception:
        return None


def clear_graph(_driver: GraphDatabase.driver) -> None:
    """Vacía completamente la base antes de cada corrida."""
    with _driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("🧹  Graph cleared.")


async def build_kg_from_docs(docs: list[Document]) -> None:
    """Construye el KG a partir de la lista de documentos proporcionada."""
    clear_graph(driver)
    ensure_property_indexes(driver)

    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        text_splitter=text_splitter,
        schema={
            "node_types": NODE_TYPES,
            "relationship_types": RELATIONSHIP_TYPES,
            "patterns": PATTERNS,
            "additional_node_types": False,
            "additional_relationship_types": False,
            "additional_patterns": False,
        },
        from_pdf=False,
    )

    # Inicializar contadores y sets para el resumen final
    summary_counts = {"chunks": 0, "presentation_date": 0}
    unique_regions: set[str] = set()
    unique_communes: set[str] = set()
    unique_tipologias: set[str] = set()
    unique_project_types: set[str] = set()
    seen_project_ids: set[int] = set()

    # Cómputos para progreso y detalle
    total_chunks = len(docs)
    # Totales por documento (source_path) = max(chunk_index) + 1
    doc_total_chunks: dict[str, int] = {}
    for d in docs:
        meta_d = d.metadata  # type: ignore[attr-defined]
        spath = str(meta_d.get("source_path"))
        cidx = int(meta_d.get("chunk_index", 0))
        doc_total_chunks[spath] = max(doc_total_chunks.get(spath, -1), cidx)
    for spath in list(doc_total_chunks.keys()):
        doc_total_chunks[spath] = doc_total_chunks[spath] + 1

    unique_projects_in_batch: set[int] = set()
    unique_docs_in_batch: set[str] = set()
    for d in docs:
        m = d.metadata  # type: ignore[attr-defined]
        try:
            unique_projects_in_batch.add(int(m.get("id")))
        except Exception:
            pass
        spath = m.get("source_path")
        if spath:
            unique_docs_in_batch.add(str(spath))

    print(
        "Iniciando carga de KG → "
        f"proyectos únicos: {len(unique_projects_in_batch)}, "
        f"documentos: {len(unique_docs_in_batch)}, "
        f"chunks: {total_chunks}"
    )

    for doc in docs:
        # Creamos una representación textual enriquecida con los metadatos para
        # facilitar la extracción por parte del LLM y construir correctamente las
        # relaciones definidas en PATTERNS.
        meta = doc.metadata  # type: ignore[attr-defined]

        # Extraemos campos relevantes desde el DataFrame de metadatos
        region = meta.get("region")
        communes_raw = meta.get("ei_document_communes") or []
        communes_list = _parse_communes(communes_raw)
        communes_str = ", ".join(communes_list)

        # Preparar variables adicionales y actualizar sets de resumen
        tipologia = meta.get("tipologia")
        project_type = meta.get("tipo_de_proyecto")
        project_name = meta.get("nombre")
        try:
            project_id_int = int(meta.get("id"))
        except Exception:
            project_id_int = -1
        source_path = str(meta.get("source_path"))
        chunk_idx = int(meta.get("chunk_index", 0))
        total_in_doc = doc_total_chunks.get(source_path, 0)

        # Progreso por chunk y encabezado de proyecto
        print(
            f"[{summary_counts['chunks'] + 1}/{total_chunks}] Proyecto {project_id_int} - {project_name}"
        )
        print(f"  Doc: {Path(source_path).name}  chunk {chunk_idx + 1}/{total_in_doc}")

        if region:
            unique_regions.add(region)
        unique_communes.update(communes_list)
        if tipologia:
            unique_tipologias.add(tipologia)
        if project_type:
            unique_project_types.add(project_type)

        augmented_text = (
            f"{doc.page_content}\n\n"  # texto original
            "---\n"
            f"Nombre del proyecto: {meta.get('nombre')}.\n"
            f"ID del proyecto: {meta.get('id')}.\n"
            f"Región: {region}.\n"
            f"Comunas involucradas: {communes_str}.\n"
            f"Tipología SEA: {meta.get('tipologia')}.\n"
            f"Tipo de proyecto: {meta.get('tipo_de_proyecto')}.\n"
            f"Fecha de presentación: {meta.get('fecha_de_presentacion')}.\n"
        )

        summary_counts["chunks"] += 1  # contar chunk procesado
        await kg_builder.run_async(text=augmented_text)

        # Aseguramos que la fecha de presentación se almacene como tipo `DATE`
        presentation_date_str = meta.get("fecha_de_presentacion")
        presentation_date_map = _iso_date_to_map(presentation_date_str)

        # -- Validaciones desactivadas, contadores actualizados para resumen --

        with driver.session() as session:
            # Asegurar existencia de Project con sus propiedades mínimas
            session.run(
                """
                MERGE (p:Project {id: $project_id})
                ON CREATE SET p.name = $project_name
                ON MATCH SET p.name = coalesce(p.name, $project_name)
                """,
                project_id=int(meta.get("id")),
                project_name=meta.get("nombre"),
            )

            if presentation_date_map:
                session.run(
                    """
                    MATCH (p:Project {id: $project_id})
                    MERGE (d:PresentationDate {date: date($fecha)})
                    MERGE (p)-[:PRESENTED_ON]->(d)
                    """,
                    project_id=int(meta.get("id")),
                    fecha=presentation_date_map,
                )
                summary_counts["presentation_date"] += 1

            # Región
            if region:
                session.run(
                    """
                    MATCH (p:Project {id: $project_id})
                    MERGE (r:Region {name: $region})
                    MERGE (p)-[:IN_REGION]->(r)
                    """,
                    project_id=int(meta.get("id")),
                    region=region,
                )

            # Comunas
            for commune in communes_list:
                session.run(
                    """
                    MATCH (p:Project {id: $project_id})
                    MERGE (c:Commune {name: $commune})
                    MERGE (p)-[:IN_COMMUNE]->(c)
                    """,
                    project_id=int(meta.get("id")),
                    commune=commune,
                )

            # Tipología
            # (comunas ya contabilizadas en unique_communes, no se requiere acción extra)
            tipologia = meta.get("tipologia")
            if tipologia:
                session.run(
                    """
                    MATCH (p:Project {id: $project_id})
                    MERGE (t:Tipologia {code: $tipologia})
                    MERGE (p)-[:HAS_TIPOLOGIA]->(t)
                    """,
                    project_id=int(meta.get("id")),
                    tipologia=tipologia,
                )

            # Tipo de proyecto
            project_type = meta.get("tipo_de_proyecto")
            if project_type:
                session.run(
                    """
                    MATCH (p:Project {id: $project_id})
                    MERGE (pt:ProjectType {name: $project_type})
                    MERGE (p)-[:HAS_PROJECT_TYPE]->(pt)
                    """,
                    project_id=int(meta.get("id")),
                    project_type=project_type,
                )

            # Tipo de documento (categoría)
            doc_type = meta.get("type")
            if doc_type:
                session.run(
                    """
                    MATCH (p:Project {id: $project_id})
                    MERGE (dt:DocumentType {name: $doc_type})
                    MERGE (p)-[:HAS_DOCUMENT_TYPE]->(dt)
                    """,
                    project_id=int(meta.get("id")),
                    doc_type=doc_type,
                )

            # Subtipo de documento (categoría)
            doc_subtype = meta.get("subtype")
            if doc_subtype:
                session.run(
                    """
                    MATCH (p:Project {id: $project_id})
                    MERGE (ds:DocumentSubtype {name: $doc_subtype})
                    MERGE (p)-[:HAS_DOCUMENT_SUBTYPE]->(ds)
                    """,
                    project_id=int(meta.get("id")),
                    doc_subtype=doc_subtype,
                )

        # Detalle por proyecto (solo primera vez con ese id)
        if project_id_int != -1 and project_id_int not in seen_project_ids:
            print(
                "  Detalle → "
                f"Región='{region}', Tipologia='{tipologia}', "
                f"Tipo='{project_type}', Subtype='{meta.get('subtype')}', "
                f"Comunas={communes_list}"
            )
            seen_project_ids.add(project_id_int)
        print(f"✅ Proyecto {meta.get('id')} listo.")

    # Imprimir resumen final
    print("🎉  Knowledge graph creation finished!")
    print("------ SUMMARY ------")
    print(f"Chunks processed: {summary_counts['chunks']}")
    print(f"Presentation dates linked: {summary_counts['presentation_date']}")
    print(f"Unique regions: {len(unique_regions)}")
    print(f"Unique communes: {len(unique_communes)}")
    print(f"Unique tipologias: {len(unique_tipologias)}")
    print(f"Unique project types: {len(unique_project_types)}")
    print("----------------------")


# --------------------------------------------------------------------------- #
# 6) Punto de entrada
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import asyncio

    async def _run():
        # Cargar todos los proyectos: concatenamos todos los chunks de todos los documentos
        all_docs: list[Document] = []
        for _k, _docs in chunk_dict.items():
            all_docs.extend(_docs)
        await build_kg_from_docs(all_docs)

    asyncio.run(_run())
