"""Main entry point for running the supervisor API module.

Run with:
    uv run -m src.API
    python -m src.API
"""

import uvicorn


def main():
    """Run the supervisor streaming API server."""
    print("=" * 70)
    print("🚀 Starting Supervisor Agent Streaming API")
    print("=" * 70)
    print("\n📍 Server: http://0.0.0.0:8000")
    print("📚 Docs:   http://0.0.0.0:8000/docs")
    print("🎨 Scalar: http://0.0.0.0:8000/scalar")
    print("\n💡 Test endpoints with curl:")
    print("   curl -X POST http://0.0.0.0:8000/ask \\")
    print("     -H 'Content-Type: application/json' \\")
    print('     -d \'{"question": "¿Cuántos proyectos hay?"}\'')
    print("\n" + "=" * 70)
    print("Press CTRL+C to stop the server")
    print("=" * 70 + "\n")

    uvicorn.run(
        "src.API.supervisor_streaming_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
