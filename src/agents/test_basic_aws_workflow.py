#!/usr/bin/env python3
"""Basic test script for AWS Neo4j integration without dependencies on embeddings.

This script tests the core functionality of the AWS Neo4j connection
for both agents without requiring OpenAI embeddings or other external services.

Usage:
    PYTHONPATH=/home/alejandro/Desktop/repos/CSW-NVIRO:$PYTHONPATH uv run src/agents/test_basic_aws_workflow.py
"""

import asyncio
import json
from typing import Any

from dotenv import load_dotenv


# Load environment variables
load_dotenv(override=True)


def test_direct_cypher_queries():
    """Test direct Cypher query execution with AWS Neo4j."""
    print("\n" + "=" * 60)
    print("🔍 Testing Direct Cypher Queries")
    print("=" * 60)

    from src.agents.cypher_query_agent.cypher_runner import run_cypher, safe_run_cypher

    test_queries = [
        {
            "name": "Count Projects",
            "query": "MATCH (p:Project) RETURN count(p) as total_projects",
        },
        {
            "name": "List Regions",
            "query": "MATCH (r:Region) RETURN r.name as region_name ORDER BY r.name LIMIT 5",
        },
        {
            "name": "Project Types",
            "query": "MATCH (pt:ProjectType) RETURN DISTINCT pt.name as type_name LIMIT 10",
        },
        {
            "name": "Projects with Locations",
            "query": """
                MATCH (p:Project)-[:IN_REGION]->(r:Region)
                RETURN p.name as project_name, r.name as region
                LIMIT 5
            """,
        },
        {
            "name": "Complex Aggregation",
            "query": """
                MATCH (p:Project)-[:IN_REGION]->(r:Region)
                RETURN r.name as region, count(p) as project_count
                ORDER BY project_count DESC
                LIMIT 5
            """,
        },
    ]

    success_count = 0
    for test in test_queries:
        print(f"\n📝 {test['name']}")
        print("-" * 40)
        try:
            # Test with run_cypher
            result = run_cypher(test["query"])
            print("✅ Query executed successfully")
            print(
                f"   Result: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}"
            )

            # Test with safe_run_cypher
            safe_result = safe_run_cypher(test["query"])
            parsed = json.loads(safe_result)
            if "error" not in parsed:
                success_count += 1
            else:
                print(f"❌ Error in safe_run_cypher: {parsed['error']}")

        except Exception as e:
            print(f"❌ Query failed: {e}")

    print(f"\n📊 Success rate: {success_count}/{len(test_queries)}")
    return success_count == len(test_queries)


def test_retriever_connection():
    """Test the retriever's connection to AWS Neo4j."""
    print("\n" + "=" * 60)
    print("🔍 Testing Retriever Connection")
    print("=" * 60)

    try:
        from src.agents.hybrid_graphRAG_agent.retriever_aws import connection, driver

        # Test basic connection
        print("Testing retriever's Neo4j connection...")
        test_results = connection.test_connection()

        if test_results["connected"]:
            print(f"✅ Retriever connected to: {test_results['uri']}")
            print(f"   Database: {test_results['database']}")
            print(f"   Server version: {test_results.get('server_version', 'Unknown')}")

            # Test a query through the driver
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) as count LIMIT 1")
                record = result.single()
                if record:
                    print(f"   Total nodes in graph: {record['count']}")

            # Test index existence
            with driver.session() as session:
                # Check for vector index
                result = session.run("SHOW INDEXES WHERE name = 'chunkEmbedding'")
                if result.single():
                    print("✅ Vector index 'chunkEmbedding' exists")

                # Check for fulltext index
                result = session.run("SHOW INDEXES WHERE name = 'chunkFulltext'")
                if result.single():
                    print("✅ Fulltext index 'chunkFulltext' exists")

            return True
        else:
            print(
                f"❌ Retriever connection failed: {test_results.get('error', 'Unknown error')}"
            )
            return False

    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
        return False


async def test_concurrent_access():
    """Test concurrent access to AWS Neo4j."""
    print("\n" + "=" * 60)
    print("🔍 Testing Concurrent Access")
    print("=" * 60)

    from src.agents.cypher_query_agent.cypher_runner import run_cypher

    async def run_query_async(query_id: int) -> dict[str, Any]:
        """Run a query asynchronously."""
        query = f"RETURN {query_id} as id, timestamp() as ts"
        try:
            result = await asyncio.to_thread(run_cypher, query)
            return {"id": query_id, "success": True, "result": result}
        except Exception as e:
            return {"id": query_id, "success": False, "error": str(e)}

    # Run 10 concurrent queries
    num_queries = 10
    print(f"Running {num_queries} concurrent queries...")

    tasks = [run_query_async(i) for i in range(num_queries)]
    results = await asyncio.gather(*tasks)

    success_count = sum(1 for r in results if r["success"])
    print(f"✅ Successful queries: {success_count}/{num_queries}")

    if success_count < num_queries:
        failed = [r for r in results if not r["success"]]
        for f in failed[:3]:  # Show first 3 failures
            print(f"   ❌ Query {f['id']} failed: {f.get('error', 'Unknown error')}")

    return success_count == num_queries


def test_schema_exploration():
    """Explore the graph schema in AWS Neo4j."""
    print("\n" + "=" * 60)
    print("🔍 Exploring Graph Schema")
    print("=" * 60)

    from src.agents.cypher_query_agent.cypher_runner import run_cypher

    try:
        # Get node labels
        labels_result = run_cypher(
            "CALL db.labels() YIELD label RETURN label ORDER BY label"
        )
        labels = [r["label"] for r in labels_result]
        print(f"\n📊 Node Labels ({len(labels)}):")
        for label in labels[:10]:  # Show first 10
            count = run_cypher(f"MATCH (n:{label}) RETURN count(n) as c")[0]["c"]
            print(f"   - {label}: {count} nodes")
        if len(labels) > 10:
            print(f"   ... and {len(labels) - 10} more labels")

        # Get relationship types
        rels_result = run_cypher(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType as type ORDER BY type"
        )
        rel_types = [r["type"] for r in rels_result]
        print(f"\n📊 Relationship Types ({len(rel_types)}):")
        for rel_type in rel_types[:10]:  # Show first 10
            print(f"   - {rel_type}")
        if len(rel_types) > 10:
            print(f"   ... and {len(rel_types) - 10} more relationship types")

        # Get some statistics
        stats = run_cypher("""
            MATCH (p:Project)
            OPTIONAL MATCH (p)-[:IN_REGION]->(r:Region)
            OPTIONAL MATCH (p)-[:IN_COMMUNE]->(c:Commune)
            RETURN
                count(DISTINCT p) as total_projects,
                count(DISTINCT r) as total_regions,
                count(DISTINCT c) as total_communes
        """)[0]

        print("\n📊 Database Statistics:")
        print(f"   - Total Projects: {stats['total_projects']}")
        print(f"   - Total Regions: {stats['total_regions']}")
        print(f"   - Total Communes: {stats['total_communes']}")

        return True

    except Exception as e:
        print(f"❌ Schema exploration failed: {e}")
        return False


async def main():
    """Main test function."""
    print("=" * 60)
    print("🚀 Basic AWS Neo4j Integration Test")
    print("=" * 60)
    print("\nTesting core AWS Neo4j functionality without external dependencies")

    # Check environment
    from src.agents.migrate_to_aws_neo4j import check_env_variables, test_aws_connection

    print("\n1️⃣ Checking environment variables...")
    if not check_env_variables():
        print("❌ Please configure your environment variables first")
        return

    print("\n2️⃣ Testing AWS Neo4j connection...")
    if not test_aws_connection():
        print("❌ Cannot connect to AWS Neo4j database")
        return

    # Run tests
    test_results = {
        "Direct Cypher Queries": test_direct_cypher_queries(),
        "Retriever Connection": test_retriever_connection(),
        "Schema Exploration": test_schema_exploration(),
        "Concurrent Access": await test_concurrent_access(),
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
        print("\n🎉 All basic tests passed!")
        print("\nThe AWS Neo4j integration is working correctly for:")
        print("  ✅ Direct Cypher query execution")
        print("  ✅ Connection pooling and concurrent access")
        print("  ✅ Retriever configuration")
        print("  ✅ Schema exploration")
        print("\n📝 Note: Full agent workflows may require additional configuration")
        print("   (e.g., OpenAI embeddings, LLM credentials)")
    else:
        print("\n⚠️ Some tests failed. Please review the errors above.")


if __name__ == "__main__":
    asyncio.run(main())
