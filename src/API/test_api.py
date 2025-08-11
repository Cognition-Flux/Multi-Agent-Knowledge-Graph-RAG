"""Test script for the Supervisor Streaming API.

This script tests all endpoints of the API to ensure they work correctly.

Run with:
    uv run python src/API/test_api.py
"""

import asyncio
import json

import aiohttp


async def test_health_check(session: aiohttp.ClientSession, base_url: str):
    """Test the health check endpoint."""
    print("\n" + "=" * 60)
    print("🧪 Testing Health Check")
    print("-" * 60)

    url = f"{base_url}/health"
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            print(f"✅ Health check passed: {data}")
        else:
            print(f"❌ Health check failed: {response.status}")


async def test_basic_streaming(session: aiohttp.ClientSession, base_url: str):
    """Test the basic streaming endpoint."""
    print("\n" + "=" * 60)
    print("🧪 Testing Basic Streaming (/ask)")
    print("-" * 60)

    url = f"{base_url}/ask"
    payload = {"question": "¿Cuántos proyectos hay en total?"}

    print(f"📝 Question: {payload['question']}")

    async with session.post(url, json=payload) as response:
        if response.status == 200:
            async for line in response.content:
                if line:
                    data = json.loads(line.decode())
                    print(f"✅ Received: {data['content'][:100]}...")
        else:
            print(f"❌ Request failed: {response.status}")


async def test_streaming_with_metadata(session: aiohttp.ClientSession, base_url: str):
    """Test streaming with metadata endpoint."""
    print("\n" + "=" * 60)
    print("🧪 Testing Streaming with Metadata (/ask-with-metadata)")
    print("-" * 60)

    url = f"{base_url}/ask-with-metadata"
    payload = {"question": "¿Qué especies están en peligro?", "include_metadata": True}

    print(f"📝 Question: {payload['question']}")

    async with session.post(url, json=payload) as response:
        if response.status == 200:
            async for line in response.content:
                if line:
                    data = json.loads(line.decode())
                    if data["type"] == "routing":
                        print(f"🎯 Routing: {data.get('metadata', {})}")
                    elif data["type"] == "answer":
                        print(f"✅ Answer: {data['content'][:100]}...")
        else:
            print(f"❌ Request failed: {response.status}")


async def test_sse_streaming(session: aiohttp.ClientSession, base_url: str):
    """Test SSE streaming endpoint."""
    print("\n" + "=" * 60)
    print("🧪 Testing SSE Streaming (/stream-sse)")
    print("-" * 60)

    url = f"{base_url}/stream-sse"
    payload = {"question": "¿En qué regiones hay proyectos?"}

    print(f"📝 Question: {payload['question']}")

    async with session.post(url, json=payload) as response:
        if response.status == 200:
            content = ""
            async for line in response.content:
                content += line.decode()

            # Parse SSE format
            lines = content.split("\n")
            for line in lines:
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        print(f"✅ SSE Data: {data['content'][:100]}...")
                    except json.JSONDecodeError:
                        pass
        else:
            print(f"❌ Request failed: {response.status}")


async def test_json_endpoint(session: aiohttp.ClientSession, base_url: str):
    """Test non-streaming JSON endpoint."""
    print("\n" + "=" * 60)
    print("🧪 Testing JSON Endpoint (/ask-json)")
    print("-" * 60)

    url = f"{base_url}/ask-json"
    payload = {
        "question": "¿Cuántas comunas tienen proyectos?",
        "include_metadata": True,
    }

    print(f"📝 Question: {payload['question']}")

    async with session.post(url, json=payload) as response:
        if response.status == 200:
            data = await response.json()
            print(f"✅ Answer: {data['answer']}")
            print(f"📊 Status: {data['status']}")
            if "routing" in data:
                print(f"🎯 Routing: {data['routing']}")
        else:
            print(f"❌ Request failed: {response.status}")
            error = await response.text()
            print(f"Error: {error}")


async def test_error_handling(session: aiohttp.ClientSession, base_url: str):
    """Test error handling with invalid requests."""
    print("\n" + "=" * 60)
    print("🧪 Testing Error Handling")
    print("-" * 60)

    url = f"{base_url}/ask"

    # Test with empty question
    print("\n📝 Testing with empty question...")
    payload = {"question": ""}

    async with session.post(url, json=payload) as response:
        if response.status == 400:
            error = await response.json()
            print(f"✅ Correctly rejected empty question: {error['detail']}")
        else:
            print(f"❌ Unexpected status: {response.status}")

    # Test with missing question field
    print("\n📝 Testing with missing question field...")
    payload = {}

    async with session.post(url, json=payload) as response:
        if response.status == 422:
            print(f"✅ Correctly rejected missing field: status {response.status}")
        else:
            print(f"❌ Unexpected status: {response.status}")


async def run_all_tests():
    """Run all API tests."""
    base_url = "http://localhost:8000"

    print("=" * 60)
    print("🚀 Starting API Tests")
    print(f"📍 Testing API at: {base_url}")
    print("=" * 60)

    # Create session with timeout
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            # Run all tests
            await test_health_check(session, base_url)
            await test_basic_streaming(session, base_url)
            await test_streaming_with_metadata(session, base_url)
            await test_sse_streaming(session, base_url)
            await test_json_endpoint(session, base_url)
            await test_error_handling(session, base_url)

            print("\n" + "=" * 60)
            print("✅ All tests completed!")
            print("=" * 60)

        except aiohttp.ClientError as e:
            print(f"\n❌ Connection error: {e}")
            print("Make sure the API is running: uv run -m src.API")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    print("Starting API test suite...")
    print("Make sure the API is running: uv run -m src.API")
    asyncio.run(run_all_tests())
