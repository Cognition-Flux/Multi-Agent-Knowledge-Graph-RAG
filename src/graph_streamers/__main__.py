"""Main entry point for running the graph_streamers package as a module.

This allows running the package with:
    uv run -m src.graph_streamers
    python -m src.graph_streamers
"""

import asyncio
import sys


async def main():
    """Main function to run the streamer demo."""
    from src.graph_streamers.async_stream_by_updates import (
        stream_supervisor_updates,
        stream_supervisor_with_metadata,
    )

    print("=" * 70)
    print("🚀 Supervisor Agent Async Streamer")
    print("=" * 70)

    # Get question from command line or use default
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\n📝 Processing question: {question}")
    else:
        # Default demo questions
        demo_questions = [
            "¿Cuántos proyectos hay en total?",
            "¿Qué especies de flora están en peligro?",
            "¿En qué regiones hay proyectos?",
        ]

        print("\n📝 Available demo questions:")
        for i, q in enumerate(demo_questions, 1):
            print(f"  {i}. {q}")

        print("\n💡 Usage examples:")
        print("  uv run -m src.graph_streamers 'Your question here'")
        print("  uv run -m src.graph_streamers.async_stream_by_updates 'Your question'")

        # Use the first demo question
        question = demo_questions[0]
        print(f"\n🎯 Using demo question: {question}")

    print("-" * 70)

    try:
        # Stream the answer
        print("\n⏳ Processing with supervisor agent...")
        print("-" * 50)

        async for answer in stream_supervisor_updates(question):
            print("\n✅ Final Answer:")
            print("=" * 50)
            print(answer)
            print("=" * 50)

        # Optional: Also show metadata
        print("\n📊 Routing Details:")
        print("-" * 50)

        async for update in stream_supervisor_with_metadata(
            question, include_routing=True
        ):
            if update["type"] == "routing":
                print(f"  🎯 Agent: {update['metadata'].get('agent', 'N/A')}")
                print(f"  💭 Reasoning: {update['metadata'].get('reasoning', 'N/A')}")
            elif update["type"] == "answer":
                print(f"  ✅ Answer length: {len(update['content'])} chars")

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ Complete!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
