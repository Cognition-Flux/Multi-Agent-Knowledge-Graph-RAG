FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

# Copy project manifests first for better layer caching
COPY pyproject.toml uv.lock ./

# Install exact dependencies into a local .venv using uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Copy application source
COPY src ./src
COPY KnowledgeGraphDB ./KnowledgeGraphDB

# Expose the FastAPI port
EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Run uvicorn via uv-managed environment
CMD ["uv", "run", "uvicorn", "src.API.supervisor_streaming_api:app", "--host", "0.0.0.0", "--port", "8000"]
