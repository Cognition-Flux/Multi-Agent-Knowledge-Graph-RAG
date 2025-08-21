from __future__ import annotations

from collections.abc import Sequence

from .list_comunas import list_comunas
from .list_comunas_en_regiones import list_comunas_en_regiones
from .list_proyectos_por_comuna_por_region import (
    list_proyectos_por_comuna_por_region,
)
from .list_regiones import list_regiones


__all__ = [
    "get_tools",
    "list_comunas",
    "list_comunas_en_regiones",
    "list_proyectos_por_comuna_por_region",
    "list_regiones",
]


def get_tools() -> Sequence[object]:
    """Return all structured tools in this package.

    The returned objects follow the LangChain tool interface and can be bound to
    LLMs via `.bind_tools()` or invoked directly via `.invoke()`.
    """
    return [
        list_regiones,
        list_comunas,
        list_comunas_en_regiones,
        list_proyectos_por_comuna_por_region,
    ]
