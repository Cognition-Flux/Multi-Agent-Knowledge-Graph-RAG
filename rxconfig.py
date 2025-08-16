"""Este fichero define la configuración *global* de Reflex para la aplicación.
Al mantenerse en la raíz del proyecto, **Reflex** lo detecta automáticamente
al arrancar y utiliza los parámetros aquí declarados —principalmente
`app_name`— para:

1. Nombrar la instancia de la aplicación (se refleja en los atributos
"""

import os
import urllib.parse as _urlparse

import reflex as rx


app_name = os.getenv("REFLEX_APP_NAME", "pathway_front")

# ---------------------------------------------------------------------------
# Dynamic URLs from environment variables
# ---------------------------------------------------------------------------
backend_url = os.getenv(
    "BACKEND_URL", "http://localhost:8000"
)  # e.g. https://xxxx.ngrok.app
frontend_domain = os.getenv(
    "FRONTEND_DOMAIN", "groker.ngrok.app"
)  # e.g. reserved ngrok domain

# Extract scheme+host (without path) for CORS if backend_url includes path
_backend_parsed = _urlparse.urlparse(backend_url)
_backend_origin = f"{_backend_parsed.scheme}://{_backend_parsed.netloc}"

config = rx.Config(
    app_name=app_name,
    frontend_port=3000,
    api_url=backend_url,
    cors_allowed_origins=[
        "http://localhost:3000",
        f"https://{frontend_domain}",
    ],
    tailwind=None,  # Explicitly disable Tailwind CSS since this project doesn't use it
    disable_plugins=[
        "reflex.plugins.sitemap.SitemapPlugin",
    ],
    vite_server_allow_hosts=[frontend_domain],
)
