#!/bin/bash
# Deploys the Reflex application using ngrok, exposing both frontend and backend.
#
# IMPORTANT: For the frontend to connect to the backend, you must configure the
# public backend URL in your Reflex app.
#
# Steps:
# 1. Run this script. ngrok will start and display public URLs.
# 2. Copy the public URL for the 'backend' tunnel (e.g., https://xxxxxxxx.ngrok.io).
# 3. Open 'rxconfig.py' and add the 'api_url' to your rx.Config:
#    config = rx.Config(
#        app_name="your_app",
#        api_url="https://xxxxxxxx.ngrok.io", # <-- Add this line
#    )
# 4. If your 'reflex run' is active, stop it (Ctrl+C) and restart it.
#    Reflex will re-compile the frontend with the correct backend URL.
# 5. Access your app using the public URL for the 'frontend' tunnel.
#
# Note: With a free ngrok account, the URL changes every time you restart ngrok.
# For a stable URL, consider a paid ngrok plan with a reserved domain.
#
# For more information on ngrok, visit: https://ngrok.com/docs

set -e

CONFIG_FILE="ngrok.yml"

# --- Main Script ---

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "Error: ngrok command not found."
    echo "Please install ngrok from https://ngrok.com/download and make sure it's in your PATH."
    exit 1
fi

# Create ngrok.yml if it doesn't exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating default configuration file: $CONFIG_FILE"
    cat > "$CONFIG_FILE" << EOL
version: "2"
# This file defines the tunnels for your Reflex application.
# Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
# and add it using the command: ngrok config add-authtoken <YOUR_TOKEN>
tunnels:
  frontend:
    proto: http
    addr: 3000
    # For a stable URL, you can reserve a domain in your ngrok dashboard
    # (e.g., my-app-frontend.ngrok.app) and uncomment the following line:
    # domain: my-app-frontend.ngrok.app
  backend:
    proto: http
    addr: 8000
    # For a stable URL, you can reserve a domain in your ngrok dashboard
    # (e.g., my-app-backend.ngrok.app) and uncomment the following line:
    # domain: my-app-backend.ngrok.app
EOL
fi

echo "Starting ngrok..."
echo "If this is your first time running ngrok, you may need to add your authtoken."
echo "You can get one from https://dashboard.ngrok.com/get-started/your-authtoken"
echo "Then run this command in another terminal: ngrok config add-authtoken <YOUR_TOKEN>"
echo ""
echo "Starting tunnels defined in $CONFIG_FILE. Press Ctrl+C to stop."
echo "----------------------------------------------------------------"

# Start all tunnels from the configuration file
ngrok start --all --config "$CONFIG_FILE"
