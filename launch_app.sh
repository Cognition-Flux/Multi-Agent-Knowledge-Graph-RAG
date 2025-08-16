#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# launch_app.sh – start Reflex *after* ngrok is already running.
# It queries ngrok’s local API (http://127.0.0.1:4040) to fetch the public
# URL that points to localhost:8000 (the backend) and uses it to set
# BACKEND_URL.  FRONTEND_DOMAIN is assumed to be the reserved domain you
# configured in ngrok.yml (edit below if you change it).
# ---------------------------------------------------------------------------
set -euo pipefail

FRONTEND_DOMAIN="groker.ngrok.app"   # <- update if you change the reserved domain
NGROK_API="http://127.0.0.1:4040/api/tunnels"

# Wait (max 10 s) for ngrok API to become reachable --------------------------------
for i in {1..10}; do
  if curl -sf "$NGROK_API" >/dev/null; then
    break
  fi
  echo "[launch_app] Waiting for ngrok API … ($i)" >&2
  sleep 1
done

# Fetch public URL whose local addr is :8000 ---------------------------------------------------
backend_url="$(
  curl -s "$NGROK_API" \
  | grep -oE '"public_url":"https:[^"]+"|"config":{"addr":"[^"]+"' \
  | paste - - \
  | grep 'localhost:8000' \
  | head -n1 \
  | sed -E 's/.*"public_url":"(https:[^"]+)".*/\1/'
)"

if [[ -z "$backend_url" ]]; then
  echo "[launch_app] ERROR: Could not detect backend tunnel (port 8000) via ngrok API." >&2
  exit 1
fi

export BACKEND_URL="$backend_url"
export FRONTEND_DOMAIN="$FRONTEND_DOMAIN"

echo "[launch_app] BACKEND_URL = $BACKEND_URL"

# Launch Reflex in the current shell so Ctrl-C stops everything.
uv run reflex run "$@"
