"""Test script for the supervisor graph streamer.

Run this script to test the async streaming functionality
without needing to set up a full API server.
"""

import asyncio

from src.graph_streamers.async_stream_by_updates import (
    stream_supervisor_updates,
    stream_supervisor_with_metadata,
)


async def test_basic_streaming():
    """Test basic streaming of supervisor responses."""
    print("=" * 70)
    print("🧪 TEST 1: Basic Streaming")
    print("=" * 70)

    test_questions = [
        # Cypher agent questions (metadata)
        "¿Cuántos proyectos hay en total?",
        "¿Qué comunas tienen proyectos?",
        # GraphRAG agent questions (content)
        "¿Qué especies de flora se encuentran en los proyectos?",
        "¿Cuáles son las especies en peligro?",
    ]

    for question in test_questions[:2]:  # Test first two questions
        print(f"\n📝 Question: {question}")
        print("-" * 50)

        try:
            # Stream the answer
            answer_received = False
            async for answer in stream_supervisor_updates(question):
                print(f"✅ Answer received (length: {len(answer)} chars)")
                print(f"📄 First 200 chars: {answer[:200]}...")
                answer_received = True

            if not answer_received:
                print("⚠️ No answer was streamed")

        except Exception as e:
            print(f"❌ Error: {e}")

        print("-" * 50)
        await asyncio.sleep(1)  # Small delay between questions


async def test_streaming_with_metadata():
    """Test streaming with routing metadata."""
    print("\n" + "=" * 70)
    print("🧪 TEST 2: Streaming with Metadata")
    print("=" * 70)

    question = "¿Qué especies de fauna están en peligro de extinción?"
    print(f"\n📝 Question: {question}")
    print("-" * 50)

    try:
        async for update in stream_supervisor_with_metadata(
            question,
            include_routing=True,
        ):
            if update["type"] == "routing":
                print("🎯 Routing Decision:")
                print(f"   - Agent: {update['metadata'].get('agent', 'unknown')}")
                print(f"   - Reasoning: {update['metadata'].get('reasoning', 'N/A')}")
            elif update["type"] == "answer":
                print("✅ Final Answer:")
                print(f"   - Length: {len(update['content'])} chars")
                print(f"   - Preview: {update['content'][:200]}...")
            elif update["type"] == "error":
                print(f"❌ Error: {update['content']}")

    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback

        traceback.print_exc()


async def test_error_handling():
    """Test error handling with invalid input."""
    print("\n" + "=" * 70)
    print("🧪 TEST 3: Error Handling")
    print("=" * 70)

    # Test with empty question
    print("\n📝 Testing with empty question...")
    try:
        async for answer in stream_supervisor_updates(""):
            print(f"Response: {answer}")
    except Exception as e:
        print(f"✅ Caught expected error: {e}")

    # Test with very long question
    print("\n📝 Testing with very long question...")
    long_question = "¿" + "test " * 500 + "?"
    try:
        async for answer in stream_supervisor_updates(long_question):
            print(f"Response length: {len(answer)} chars")
            break  # Just check if it starts processing
    except Exception as e:
        print(f"Error: {e}")


async def test_concurrent_streaming():
    """Test concurrent streaming of multiple questions."""
    print("\n" + "=" * 70)
    print("🧪 TEST 4: Concurrent Streaming")
    print("=" * 70)

    questions = [
        "¿Cuántos proyectos hay?",
        "¿Qué especies de flora existen?",
        "¿En qué regiones hay proyectos?",
    ]

    async def stream_question(idx: int, question: str):
        """Stream a single question and return result."""
        try:
            print(f"[{idx}] Starting: {question}")
            async for answer in stream_supervisor_updates(question):
                print(f"[{idx}] ✅ Received answer ({len(answer)} chars)")
                return answer
        except Exception as e:
            print(f"[{idx}] ❌ Error: {e}")
            return None

    # Run all questions concurrently
    tasks = [stream_question(i, q) for i, q in enumerate(questions)]

    print("\n🚀 Starting concurrent streams...")
    results = await asyncio.gather(*tasks)

    print("\n📊 Results summary:")
    for i, (question, result) in enumerate(zip(questions, results, strict=False)):
        if result:
            print(f"[{i}] ✅ Success - {len(result)} chars")
        else:
            print(f"[{i}] ❌ Failed")


async def main():
    """Run all tests."""
    print("🚀 Starting Supervisor Streamer Tests")
    print("=" * 70)

    # Run tests sequentially
    await test_basic_streaming()
    await test_streaming_with_metadata()
    await test_error_handling()
    await test_concurrent_streaming()

    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    # Run with: python src/graph_streamers/test_streamer.py
    # Or: uv run python src/graph_streamers/test_streamer.py
    asyncio.run(main())
