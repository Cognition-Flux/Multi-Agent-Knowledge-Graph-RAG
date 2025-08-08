"""Módulo para gestionar la conexión a la BD."""

# %%
import os

import psycopg
from dotenv import load_dotenv


load_dotenv(override=True)


def get_conn():
    """Abre una conexión a la BD usando variables de entorno.

    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD.
    """
    load_dotenv(override=True)  # lee .env si existe
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )
