#
# Web Interface                 http://127.0.0.1:4040
# Forwarding                    https://97bb3fdb48a6.ngrok.app -> http://localhost:8000
# Forwarding                    https://groker.ngrok.app -> http://localhost:3000

#
export BACKEND_URL=https://97bb3fdb48a6.ngrok.app          # copy from ngrok
export FRONTEND_DOMAIN=groker.ngrok.app
uv run reflex run
