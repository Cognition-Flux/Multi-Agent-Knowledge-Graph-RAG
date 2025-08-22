### Run with Docker Compose (recommended)

Prerequisites: Docker and Docker Compose installed.

1) Build and start in background

```bash
docker compose up --build -d
```

2) Follow logs

```bash
docker compose logs -f
```

3) Open in browser

- UI: `http://localhost:3000`
- Backend: `http://localhost:8000`

4) Stop services

```bash
docker compose down
```

Note: Hot reload is enabled via bind mount; dependencies persist in a named volume at `/app/.venv`.

### Run with plain Docker (optional)

Build the image:

```bash
docker build -t csw-nviro-app .
```

Run the container:

```bash
docker run --name csw-nviro-app \
  -p 3000:3000 -p 8000:8000 \
  -e WATCHFILES_FORCE_POLLING=true \
  -v "$(pwd)":/app \
  -v csw_nviro_venv:/app/.venv \
  csw-nviro-app
```

Then visit:

- UI: `http://localhost:3000`
- Backend: `http://localhost:8000`
