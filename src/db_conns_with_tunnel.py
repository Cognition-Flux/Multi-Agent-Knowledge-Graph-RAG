"""Módulo para gestionar la conexión a la BD con soporte para SSH tunnel."""

import os
from contextlib import contextmanager
from urllib.parse import quote_plus

import psycopg
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sshtunnel import SSHTunnelForwarder


load_dotenv(override=True)


def get_conn():
    """Abre una conexión directa a la BD usando variables de entorno.

    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD.

    Nota: Esta función asume que la BD está accesible directamente (ej: localhost).
    Para conexiones remotas via SSH, usar get_tunnel_conn() o get_tunnel_engine().
    """
    load_dotenv(override=True)  # lee .env si existe
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )


@contextmanager
def get_tunnel_conn():
    """Abre una conexión a la BD a través de SSH tunnel usando psycopg.

    Usa las variables de entorno SSH_* y POSTGRES_* definidas en .env.

    Ejemplo:
        with get_tunnel_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                print(cur.fetchone())
    """
    load_dotenv(override=True)

    # SSH Configuration
    ssh_host = os.environ["SSH_HOST"]
    ssh_user = os.environ["SSH_USER"]
    ssh_port = int(os.getenv("SSH_PORT", "22"))
    ssh_key = os.environ["SSH_PKEY_PATH"]

    # Remote DB endpoint
    remote_host = os.environ["TUNNEL_REMOTE_BIND_HOST"]
    remote_port = int(os.getenv("TUNNEL_REMOTE_BIND_PORT", "5432"))

    # DB Credentials
    db_name = os.environ["POSTGRES_DB"]
    db_user = os.environ["POSTGRES_USER"]
    db_pwd = os.environ["POSTGRES_PASSWORD"]

    # Create SSH tunnel
    tunnel = SSHTunnelForwarder(
        (ssh_host, ssh_port),
        ssh_username=ssh_user,
        ssh_pkey=ssh_key,
        remote_bind_address=(remote_host, remote_port),
        local_bind_address=("127.0.0.1", 0),
    )

    try:
        tunnel.start()

        # Create psycopg connection through tunnel
        conn = psycopg.connect(
            host="127.0.0.1",
            port=tunnel.local_bind_port,
            dbname=db_name,
            user=db_user,
            password=db_pwd,
        )

        yield conn

    finally:
        if "conn" in locals():
            conn.close()
        tunnel.stop()


def get_tunnel_engine() -> tuple[SSHTunnelForwarder, Engine]:
    """Crea un túnel SSH y un Engine de SQLAlchemy.

    Retorna una tupla (tunnel, engine). El llamador es responsable de cerrar ambos.

    Ejemplo:
        tunnel, engine = get_tunnel_engine()
        try:
            df = pd.read_sql("SELECT * FROM table", engine)
        finally:
            engine.dispose()
            tunnel.stop()
    """
    load_dotenv(override=True)

    # SSH Configuration
    ssh_host = os.environ["SSH_HOST"]
    ssh_user = os.environ["SSH_USER"]
    ssh_port = int(os.getenv("SSH_PORT", "22"))
    ssh_key = os.environ["SSH_PKEY_PATH"]

    # Remote DB endpoint
    remote_host = os.environ["TUNNEL_REMOTE_BIND_HOST"]
    remote_port = int(os.getenv("TUNNEL_REMOTE_BIND_PORT", "5432"))

    # DB Credentials
    db_name = os.environ["POSTGRES_DB"]
    db_user = os.environ["POSTGRES_USER"]
    db_pwd = os.environ["POSTGRES_PASSWORD"]

    # Create SSH tunnel
    tunnel = SSHTunnelForwarder(
        (ssh_host, ssh_port),
        ssh_username=ssh_user,
        ssh_pkey=ssh_key,
        remote_bind_address=(remote_host, remote_port),
        local_bind_address=("127.0.0.1", 0),
    )
    tunnel.start()

    # Build database URL with URL-encoded password
    db_pwd_encoded = quote_plus(db_pwd)
    db_url = (
        f"postgresql+psycopg2://{db_user}:{db_pwd_encoded}@"
        f"127.0.0.1:{tunnel.local_bind_port}/{db_name}"
    )

    # Create SQLAlchemy engine
    engine = create_engine(db_url, pool_pre_ping=True)

    return tunnel, engine


def is_local_db_available() -> bool:
    """Verifica si hay una BD local disponible (ej: Docker)."""
    try:
        conn = get_conn()
        conn.close()
        return True
    except Exception:
        return False


def get_smart_conn(
    prefer_tunnel: bool = False,
) -> psycopg.Connection | tuple[SSHTunnelForwarder, Engine]:
    """Obtiene una conexión inteligente: local si está disponible, túnel SSH si no.

    Args:
        prefer_tunnel: Si True, siempre usa túnel SSH. Si False, intenta local primero.

    Returns:
        Si es local: psycopg.Connection
        Si es túnel: tuple(SSHTunnelForwarder, Engine)
    """
    if not prefer_tunnel and is_local_db_available():
        print("📍 Usando conexión local a la base de datos")
        return get_conn()
    else:
        print("🔐 Usando conexión via túnel SSH")
        return get_tunnel_engine()
