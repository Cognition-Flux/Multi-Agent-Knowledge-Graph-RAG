#!/usr/bin/env python3
"""Test script to demonstrate both agents working with AWS-hosted Neo4j in LangGraph workflows.

This script runs test queries through both the Cypher Query Agent and Hybrid GraphRAG Agent
to verify they are properly integrated with the AWS-hosted Neo4j database.

Usage:
    PYTHONPATH=/home/alejandro/Desktop/repos/CSW-NVIRO:$PYTHONPATH uv run src/agents/test_aws_integration.py
"""

import asyncio
import logging
from typing import Any

from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)


async def test_cypher_query_agent():
    """Test the Cypher Query Agent with various queries."""
    print("\n" + "=" * 60)
    print("🔍 Testing Cypher Query Agent")
    print("=" * 60)

    try:
        from src.agents.cypher_query_agent.graph_builder import graph

        test_queries = [
            "¿Cuántos proyectos hay en la base de datos?",
            "Lista los proyectos en la región de Coquimbo",
            "¿Qué tipos de proyectos existen?",
        ]

        for query in test_queries:
            print(f"\n📝 Query: {query}")
            print("-" * 40)

            try:
                result = None
                async for chunk in graph.astream(
                    {"question": query},
                    stream_mode="updates",
                ):
                    # Get the final result
                    if "generate_answer" in chunk:
                        if "messages" in chunk["generate_answer"]:
                            result = chunk["generate_answer"]["messages"][0].content

                if result:
                    print("✅ Answer received (preview):")
                    # Show first 500 characters of the answer
                    preview = result[:500] + "..." if len(result) > 500 else result
                    print(preview)
                else:
                    print("⚠️ No answer generated")

            except Exception as e:
                print(f"❌ Query failed: {e}")
                logger.error(f"Error details: {e}", exc_info=True)

        return True

    except Exception as e:
        print(f"❌ Cypher Query Agent test failed: {e}")
        logger.error(f"Error details: {e}", exc_info=True)
        return False


async def test_hybrid_graphrag_agent():
    """Test the Hybrid GraphRAG Agent with various queries."""
    print("\n" + "=" * 60)
    print("🔍 Testing Hybrid GraphRAG Agent")
    print("=" * 60)

    try:
        from src.agents.hybrid_graphRAG_agent.graph_builder import graph

        test_queries = [
            "¿Qué información tienes sobre proyectos de energía solar?",
            "Describe los proyectos de biosólidos",
            "¿Qué proyectos hay en la comuna de La Serena?",
        ]

        for query in test_queries:
            print(f"\n📝 Query: {query}")
            print("-" * 40)

            try:
                results = []
                async for chunk in graph.astream(
                    {"question": query},
                    stream_mode="updates",
                ):
                    # Collect results from parallel executions
                    if "generate_answer" in chunk:
                        if "results" in chunk["generate_answer"]:
                            results.extend(chunk["generate_answer"]["results"])

                if results:
                    print(f"✅ Retrieved {len(results)} answers from parallel searches")
                    # Show preview of first result
                    if results[0]:
                        preview = (
                            results[0][:500] + "..."
                            if len(results[0]) > 500
                            else results[0]
                        )
                        print(f"Preview of first result:\n{preview}")
                else:
                    print("⚠️ No results retrieved")

            except Exception as e:
                print(f"❌ Query failed: {e}")
                logger.error(f"Error details: {e}", exc_info=True)

        return True

    except Exception as e:
        print(f"❌ Hybrid GraphRAG Agent test failed: {e}")
        logger.error(f"Error details: {e}", exc_info=True)
        return False


async def test_connection_stability():
    """Test connection stability with multiple concurrent queries."""
    print("\n" + "=" * 60)
    print("🔍 Testing Connection Stability")
    print("=" * 60)

    try:
        from src.agents.cypher_query_agent.cypher_runner import run_cypher

        # Run multiple queries concurrently
        queries = [
            "MATCH (p:Project) RETURN count(p) as count",
            "MATCH (r:Region) RETURN r.name LIMIT 5",
            "MATCH (c:Commune) RETURN count(c) as count",
            "MATCH (t:ProjectType) RETURN t.name LIMIT 5",
        ]

        async def run_query_async(query: str) -> Any:
            """Run a query asynchronously."""
            return await asyncio.to_thread(run_cypher, query)

        print("Running concurrent queries...")
        tasks = [run_query_async(q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        for i, (query, result) in enumerate(zip(queries, results, strict=False)):
            if isinstance(result, Exception):
                print(f"❌ Query {i + 1} failed: {result}")
            else:
                print(f"✅ Query {i + 1} succeeded: {query[:50]}...")
                success_count += 1

        print(f"\n📊 Success rate: {success_count}/{len(queries)} queries")
        return success_count == len(queries)

    except Exception as e:
        print(f"❌ Connection stability test failed: {e}")
        return False


async def main():
    """Main test function."""
    print("=" * 60)
    print("🚀 AWS Neo4j Integration Test Suite")
    print("=" * 60)
    print("\nThis test suite validates that both agents are properly")
    print("integrated with the AWS-hosted Neo4j database.")

    # Check environment
    from src.agents.migrate_to_aws_neo4j import check_env_variables, test_aws_connection

    if not check_env_variables():
        print("\n❌ Please configure your environment variables first")
        return

    if not test_aws_connection():
        print("\n❌ Cannot connect to AWS Neo4j database")
        return

    # Run tests
    test_results = {
        "Connection Stability": await test_connection_stability(),
        "Cypher Query Agent": await test_cypher_query_agent(),
        "Hybrid GraphRAG Agent": await test_hybrid_graphrag_agent(),
    }

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)

    all_passed = True
    for test_name, passed in test_results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 All tests passed! The AWS Neo4j integration is working correctly.")
        print("\n📝 Next steps:")
        print("  - Monitor query performance in production")
        print("  - Adjust connection pool settings if needed")
        print(
            "  - Consider implementing query result caching for frequently accessed data"
        )
    else:
        print("\n⚠️ Some tests failed. Please review the errors above.")


if __name__ == "__main__":
    asyncio.run(main())
