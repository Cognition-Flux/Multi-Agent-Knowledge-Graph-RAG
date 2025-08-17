"""Este script crea un Knowledge Graph (KG) en Neo4j a partir de chunks de texto.

uv run -m KnowledgeGraphDB.Neo4j_KG_creation.knowledge_graph_builder
"""

# %%
from __future__ import annotations

import asyncio
import contextlib

# Utilidades de robustez
import hashlib
import os
import re
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv
from langchain_core.documents import Document
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.cohere import CohereEmbeddings
from neo4j_graphrag.indexes import (
    create_fulltext_index,
    create_vector_index,
)

from src.config import CHUNKS_REFINED_COLLECTION_DIR
from src.documents.markdown_chunking_step02 import load_chunks_from_file


load_dotenv(override=True)

NEO4J_USERNAME: str | None = os.getenv("NEO4J_USERNAME_UPGRADED")
NEO4J_PASSWORD: str | None = os.getenv("NEO4J_PASSWORD_UPGRADED")
NEO4J_URI: str | None = os.getenv("NEO4J_CONNECTION_URI_UPGRADED")

if not (NEO4J_USERNAME and NEO4J_PASSWORD and NEO4J_URI):
    raise OSError("⚠️  Variables de entorno de Neo4j incompletas. Revisa `.env`.")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
# Verificamos conectividad antes de proseguir sin cerrar el driver
driver.verify_connectivity()


def _try_create_embedder() -> Any | None:
    """Crea un embedder si hay credenciales; si no, retorna None para fallback sin embedding."""
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        return None
    try:
        return CohereEmbeddings(model="embed-v4.0", api_key=api_key)
    except Exception:
        return None


embedder = _try_create_embedder()

# --------------------------------------------------------------------------- #
# 2.1) Crear índices vectoriales y full-text para los nodos Chunk
# --------------------------------------------------------------------------- #

VECTOR_INDEX_NAME = "chunkEmbedding"
FULLTEXT_INDEX_NAME = "chunkFulltext"

# Intentamos inferir la dimensión automáticamente.
if embedder is not None:
    try:
        VECTOR_DIM = len(embedder.embed_query("test"))
    except Exception:
        VECTOR_DIM = 1024  # fallback razonable para Cohere v4
else:
    # Sin embedder, usamos 1024 para ser compatible con Cohere v4 si luego se habilita
    VECTOR_DIM = 1024

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
# Naming helpers for new UUID-based chunk files                               #
# --------------------------------------------------------------------------- #

_UUID_REFINED_JSONL_RE: re.Pattern = re.compile(
    r"^[0-9a-f]{8}_augmented\.jsonl$", re.IGNORECASE
)

# --------------------------------------------------------------------------- #
# 2.2) Restaurar chunks agrupados por documento
# --------------------------------------------------------------------------- #


def _collect_refined_jsonl_files() -> list[Path]:
    """Return list of refined JSONL files produced by *markdown_chunking_step02*.

    The preferred naming scheme is ``<uuid>_augmented.jsonl`` where *uuid* is the
    8-character hexadecimal identifier of the originating document.  If no files
    match that pattern (e.g., legacy runs), we fall back to every ``*.jsonl`` in
    the directory for backward compatibility.
    """
    directory = Path(CHUNKS_REFINED_COLLECTION_DIR)
    if not directory.exists():
        raise FileNotFoundError(
            f"Directorio {directory} no existe. Ejecuta el script step02 para generar chunks refinados."
        )

    files = sorted(
        p
        for p in directory.glob("*.jsonl")
        if p.is_file() and _UUID_REFINED_JSONL_RE.match(p.name)
    )

    if not files:
        print(
            "! WARNING: No refined JSONL files matched the UUID pattern. "
            "Using every *.jsonl file in the directory as fallback."
        )
        files = sorted(p for p in directory.glob("*.jsonl") if p.is_file())

    return files


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

# --------------------------------------------------------------------------- #
# Naming helpers for new UUID-based chunk files                               #
# --------------------------------------------------------------------------- #


def ensure_property_indexes(_driver: GraphDatabase.driver) -> None:
    """Ensure minimal constraints and indexes without destructive operations."""
    with _driver.session() as session:
        # Create unique constraints (idempotent)
        session.run(
            "CREATE CONSTRAINT project_id_unique IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE"
        )
        session.run(
            "CREATE CONSTRAINT chunk_uid_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.uid IS UNIQUE"
        )

        # Non-unique supporting indexes (IF NOT EXISTS is safe and non-destructive)
        session.run(
            "CREATE INDEX project_name_idx IF NOT EXISTS FOR (p:Project) ON (p.name)"
        )
        session.run(
            "CREATE INDEX region_name_idx IF NOT EXISTS FOR (r:Region) ON (r.name)"
        )
        session.run(
            "CREATE INDEX commune_name_idx IF NOT EXISTS FOR (c:Commune) ON (c.name)"
        )
        session.run(
            "CREATE INDEX tipologia_code_idx IF NOT EXISTS FOR (t:Tipologia) ON (t.code)"
        )
        session.run(
            "CREATE INDEX projecttype_name_idx IF NOT EXISTS FOR (pt:ProjectType) ON (pt.name)"
        )
        session.run(
            "CREATE INDEX doctype_name_idx IF NOT EXISTS FOR (dt:DocumentType) ON (dt.name)"
        )
        session.run(
            "CREATE INDEX docsubtype_name_idx IF NOT EXISTS FOR (ds:DocumentSubtype) ON (ds.name)"
        )
        session.run(
            "CREATE INDEX presentation_date_idx IF NOT EXISTS FOR (d:PresentationDate) ON (d.date)"
        )


# ---------------------------- Utilidades de robustez ------------------------- #
T = TypeVar("T")


def _with_retry(
    func: Callable[[], T],
    *,
    retries: int = 5,
    base_delay_s: float = 0.5,
    max_delay_s: float = 4.0,
    retry_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Ejecuta `func` con reintentos exponenciales en errores recuperables."""
    attempt = 0
    while True:
        try:
            return func()
        except retry_exceptions:
            attempt += 1
            if attempt > retries:
                raise
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
            time.sleep(delay)


def _compute_chunk_uid(source_path: str, chunk_index: int) -> str:
    raw = f"{source_path}|{chunk_index}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _safe_int(value: Any, fallback: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


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


def clear_graph(_driver: GraphDatabase.driver) -> None:  # legacy
    """Deprecated: ya no limpiamos la base al iniciar."""
    return


async def build_kg_from_docs(docs: list[Document]) -> None:
    """Construye el KG a partir de la lista de documentos proporcionada."""
    ensure_property_indexes(driver)

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
        # cadena amigable solo para logs: no se usa en DB

        # Preparar variables adicionales y actualizar sets de resumen
        tipologia = meta.get("tipologia")
        project_type = meta.get("tipo_de_proyecto")
        project_name = meta.get("nombre")
        project_id_int = _safe_int(meta.get("id"), -1)
        source_path_val = meta.get("source_path")
        if not source_path_val or not isinstance(source_path_val, str):
            print("  ⚠️  Chunk sin 'source_path' válido. Se omite.")
            continue
        source_path = source_path_val
        chunk_idx = _safe_int(meta.get("chunk_index", 0), 0)
        total_in_doc = doc_total_chunks.get(source_path, 0)
        chunk_uid = _compute_chunk_uid(source_path, chunk_idx)

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

        # 1) Crear el Chunk si no existe (idempotente)
        def _ensure_chunk(
            uid: str = chunk_uid,
            page_text: str = doc.page_content,
            idx: int = chunk_idx,
            src: str = source_path,
            heading: str | None = meta.get("h1"),
        ) -> bool:
            with driver.session() as session:
                record = session.run(
                    (
                        "MATCH (c:Chunk) WHERE c.uid = $uid "
                        "OR (c.source_path = $src AND c.chunk_index = $idx) "
                        "RETURN c LIMIT 1"
                    ),
                    {"uid": uid, "src": src, "idx": idx},
                ).single()
                if record is not None and record.get("c") is not None:
                    # Normalizar: asegurar uid esté seteado
                    session.run(
                        (
                            "MATCH (c:Chunk) WHERE id(c) = $nid "
                            "SET c.uid = coalesce(c.uid, $uid)"
                        ),
                        {"nid": record["c"].id, "uid": uid},
                    )
                    return False
                session.run(
                    (
                        "CREATE (c:Chunk {uid: $uid, text: $text, chunk_index: $chunk_index, "
                        "source_path: $source_path, h1: $h1})"
                    ),
                    {
                        "uid": uid,
                        "text": page_text,
                        "chunk_index": idx,
                        "source_path": src,
                        "h1": heading,
                    },
                )
                return True

        created = _with_retry(_ensure_chunk)

        # Si el chunk ya existía, lo omitimos para evitar duplicados y sobrecarga innecesaria
        if not created:
            print(
                f"⚠️  Chunk duplicado detectado (uid={chunk_uid}). Se omite procesamiento."
            )
            continue

        # 2) Embedding del chunk y seteo de propiedad para vector index (best-effort)
        if embedder is not None and created:
            with contextlib.suppress(Exception):
                embedding_vector = embedder.embed_query(doc.page_content)

                def _set_embedding(
                    uid: str = chunk_uid, emb: list[float] = embedding_vector
                ) -> None:
                    with driver.session() as session:
                        session.run(
                            "MATCH (c:Chunk {uid: $uid}) SET c.embedding = $embedding",
                            {"uid": uid, "embedding": emb},
                        )

                _with_retry(_set_embedding)

        # Aseguramos que la fecha de presentación se almacene como tipo `DATE`
        presentation_date_str = meta.get("fecha_de_presentacion")
        presentation_date_map = _iso_date_to_map(presentation_date_str)

        # 3) Relaciones al Project y taxonomías (si hay project_id válido)
        if project_id_int != -1:

            def _ensure_project_and_links(
                uid: str = chunk_uid,
                pid: int = project_id_int,
                pname: str | None = project_name,
                pres_date: dict[str, int] | None = presentation_date_map,
                reg: str | None = region,
                communes: list[str] = communes_list,
                tip: str | None = tipologia,
                ptype: str | None = project_type,
                doctype: str | None = meta.get("type"),
                docsub: str | None = meta.get("subtype"),
            ) -> None:
                with driver.session() as session:
                    # Project básico y relación HAS_CHUNK
                    session.run(
                        (
                            "MATCH (c:Chunk {uid: $uid}) "
                            "MERGE (p:Project {id: $project_id}) "
                            "ON CREATE SET p.name = $project_name "
                            "MERGE (p)-[:HAS_CHUNK]->(c)"
                        ),
                        {"uid": uid, "project_id": pid, "project_name": pname},
                    )

                    # Fecha de presentación
                    if pres_date:
                        session.run(
                            (
                                "MATCH (p:Project {id: $project_id}) "
                                "MERGE (d:PresentationDate {date: date($fecha)}) "
                                "MERGE (p)-[:PRESENTED_ON]->(d)"
                            ),
                            {"project_id": pid, "fecha": pres_date},
                        )

                    # Región
                    if reg:
                        session.run(
                            (
                                "MATCH (p:Project {id: $project_id}) "
                                "MERGE (r:Region {name: $region}) "
                                "MERGE (p)-[:IN_REGION]->(r)"
                            ),
                            {"project_id": pid, "region": reg},
                        )

                    # Comunas
                    for commune in communes:
                        session.run(
                            (
                                "MATCH (p:Project {id: $project_id}) "
                                "MERGE (c:Commune {name: $commune}) "
                                "MERGE (p)-[:IN_COMMUNE]->(c)"
                            ),
                            {"project_id": pid, "commune": commune},
                        )

                    # Tipología
                    if tip:
                        session.run(
                            (
                                "MATCH (p:Project {id: $project_id}) "
                                "MERGE (t:Tipologia {code: $tipologia}) "
                                "MERGE (p)-[:HAS_TIPOLOGIA]->(t)"
                            ),
                            {"project_id": pid, "tipologia": tip},
                        )

                    # Tipo de proyecto
                    if ptype:
                        session.run(
                            (
                                "MATCH (p:Project {id: $project_id}) "
                                "MERGE (pt:ProjectType {name: $project_type}) "
                                "MERGE (p)-[:HAS_PROJECT_TYPE]->(pt)"
                            ),
                            {"project_id": pid, "project_type": ptype},
                        )

                    # Tipo de documento (categoría)
                    if doctype:
                        session.run(
                            (
                                "MATCH (p:Project {id: $project_id}) "
                                "MERGE (dt:DocumentType {name: $doc_type}) "
                                "MERGE (p)-[:HAS_DOCUMENT_TYPE]->(dt)"
                            ),
                            {"project_id": pid, "doc_type": doctype},
                        )

                    # Subtipo de documento (categoría)
                    if docsub:
                        session.run(
                            (
                                "MATCH (p:Project {id: $project_id}) "
                                "MERGE (ds:DocumentSubtype {name: $doc_subtype}) "
                                "MERGE (p)-[:HAS_DOCUMENT_SUBTYPE]->(ds)"
                            ),
                            {"project_id": pid, "doc_subtype": docsub},
                        )

            _with_retry(_ensure_project_and_links)
            if presentation_date_map:
                summary_counts["presentation_date"] += 1

        # Detalle por proyecto (solo primera vez con ese id)
        if project_id_int != -1 and project_id_int not in seen_project_ids:
            print(
                "  Detalle → "
                f"Región='{region}', Tipologia='{tipologia}', "
                f"Tipo='{project_type}', Subtype='{meta.get('subtype')}', "
                f"Comunas={communes_list}"
            )
            seen_project_ids.add(project_id_int)
        summary_counts["chunks"] += 1
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

    async def _run():
        # Cargar todos los proyectos: concatenamos todos los chunks de todos los documentos
        all_docs: list[Document] = []
        for _k, _docs in chunk_dict.items():
            all_docs.extend(_docs)
        await build_kg_from_docs(all_docs)

    asyncio.run(_run())
