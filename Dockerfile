# Use an official Python runtime as a parent image
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system tools required for building Python packages with native extensions
# - curl & unzip: for downloading/unpacking toolchains (e.g., Node downloaded by Reflex)
# - build-essential: provides gcc/g++ & make for compiling C/C++ extensions
# - git: some Python packages fetch sub-modules during build
# NOTE: We remove the apt lists afterwards to keep the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        unzip \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv using the official installer script
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# Add uv to PATH
ENV PATH="/root/.local/bin:${PATH}"

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml ./
COPY uv.lock ./

# Install project dependencies into a local virtual environment
RUN uv sync --frozen

# Copy the rest of the application code
COPY . .

# Expose ports (Reflex default: 3000 for frontend, 8000 for backend)
EXPOSE 3000
EXPOSE 8000

# Default command to run the app using uv
# Reflex will serve the backend and frontend; the backend binds to 0.0.0.0 in containers
CMD ["uv", "run", "reflex", "run"]
