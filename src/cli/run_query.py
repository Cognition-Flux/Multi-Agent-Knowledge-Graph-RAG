# src/cli/run_query.py
#!/usr/bin/env python
"""Ejecuta la consulta SQL `sql/flora_fauna_query.sql`.

muestra un resumen

por pantalla y guarda el resultado como Parquet comprimido en el mismo
directorio (`src/cli/flora_fauna_metadata.parquet`).

Uso:
    uv run -m src.cli.run_query
"""

# %%
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.db_conns import get_conn


load_dotenv(override=True)


def read_sql() -> str:
    """Devuelve el contenido de sql/flora_fauna_query.sql."""
    sql_path = (
        Path(__file__).resolve().parents[2]  # -> proyecto raíz
        / "sql"
        / "flora_fauna_query.sql"
    )
    return sql_path.read_text(encoding="utf-8")


def main():
    query = read_sql()

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(query)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=columns)
    print(df.head())  # muestra las primeras filas

    output_file = Path(__file__).parent / "flora_fauna_metadata.parquet"
    df.to_parquet(output_file, index=False, compression="gzip")
    print(f"Resultado Parquet (comprimido) guardado en {output_file}")


if __name__ == "__main__":
    main()
