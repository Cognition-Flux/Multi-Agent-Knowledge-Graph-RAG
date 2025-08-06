"""Utility functions to load flora and fauna metadata.

This module provides a convenience wrapper that loads the Parquet file
created by `src.cli.run_query` into a pandas ``DataFrame``.  It keeps
all file-system logic in one place so that notebooks and application
code can simply do::

    from src.documents.metadata import load_metadata

    df = load_metadata()

The loader works out-of-the-box as long as the Parquet file lives at the
canonical location ``src/cli/flora_fauna_metadata.parquet``.  If you
have stored the dataset elsewhere you can pass the path explicitly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Default path for flora & fauna metadata Parquet file (imported from config)
from src.config import FLORA_FAUNA_PARQUET_PATH as _DEFAULT_PARQUET_PATH


def load_metadata(path: str | Path | None = None, **read_kwargs) -> pd.DataFrame:
    """Return a :class:`~pandas.DataFrame` with flora & fauna metadata.

    Parameters
    ----------
    path
        Optional filesystem path to the Parquet file.  When *None* (the
        default) the canonical location relative to the repository root
        is used.
    **read_kwargs
        Additional keyword arguments forwarded verbatim to
        :func:`pandas.read_parquet`.  This lets callers tweak the engine,
        columns to load, et cetera.

    Returns:
    -------
    pandas.DataFrame
        The metadata as loaded from Parquet.
    """
    parquet_path: Path = Path(path) if path is not None else _DEFAULT_PARQUET_PATH

    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el fichero Parquet en {parquet_path!s}. "
            "Ejecuta primero `uv run -m src.cli.run_query` para generarlo "
            "o pasa la ruta manualmente."
        )

    return pd.read_parquet(parquet_path, **read_kwargs)


__all__ = ["load_metadata"]
