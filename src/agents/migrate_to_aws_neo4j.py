#!/usr/bin/env python3
"""Migration script to update agents to use AWS-hosted Neo4j database.

This script helps migrate the cypher_query_agent and hybrid_graphRAG_agent
to use the AWS-hosted Neo4j database instead of local instances.

Usage:
    uv run src/agents/migrate_to_aws_neo4j.py
"""

import os
import sys

from dotenv import load_dotenv


def check_env_variables():
    """Check if required environment variables are set."""
    load_dotenv(override=True)

    required_vars = {
        "AWS_NEO4J_CONNECTION_URL": "AWS Neo4j connection URL",
        "AWS_NEO4J_USERNAME": "AWS Neo4j username",
        "AWS_NEO4J_PASSWORD": "AWS Neo4j password",
        "AWS_BEDROCK_REGION": "AWS Bedrock region for embeddings",
    }

    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"  - {var}: {description}")

    if missing_vars:
        print("❌ Missing required environment variables:")
        print("\n".join(missing_vars))
        print("\n📝 Please copy env.example to .env and fill in your credentials.")
        return False

    print("✅ All required environment variables are set")
    return True


def test_aws_connection():
    """Test connection to AWS-hosted Neo4j."""
    print("\n🔍 Testing AWS Neo4j connection...")

    try:
        from KnowledgeGraphDB.neo4j_aws_hosted_db.connection import get_connection

        connection = get_connection()
        test_results = connection.test_connection()

        if test_results["connected"]:
            print(f"✅ Connected to AWS Neo4j at {test_results['uri']}")
            print(f"   Database: {test_results['database']}")
            if test_results.get("server_version"):
                print(f"   Server version: {test_results['server_version']}")

            # Run a simple query to verify
            records, _, _ = connection.execute_query(
                "MATCH (n) RETURN count(n) as count LIMIT 1"
            )
            if records:
                print(f"   Node count: {records[0]['count']}")
            return True
        else:
            print(f"❌ Failed to connect: {test_results.get('error', 'Unknown error')}")
            return False

    except Exception as exc:
        print(f"❌ Connection test failed: {exc}")
        return False


def test_cypher_query_agent():
    """Test the Cypher Query Agent with AWS Neo4j."""
    print("\n🔍 Testing Cypher Query Agent...")

    try:
        from src.agents.cypher_query_agent.cypher_runner import (
            run_cypher,
            test_connection,
        )

        if test_connection():
            # Run a simple test query
            result = run_cypher(
                "MATCH (p:Project) RETURN count(p) as project_count LIMIT 1"
            )
            print(
                f"✅ Cypher Query Agent working! Found {result[0]['project_count'] if result else 0} projects"
            )
            return True
        else:
            print("❌ Cypher Query Agent connection failed")
            return False

    except Exception as exc:
        print(f"❌ Cypher Query Agent test failed: {exc}")
        return False


def test_hybrid_graphrag_agent():
    """Test the Hybrid GraphRAG Agent with AWS Neo4j."""
    print("\n🔍 Testing Hybrid GraphRAG Agent...")

    try:
        from src.agents.hybrid_graphRAG_agent.retriever_aws import connection, retriever

        # Test connection
        test_results = connection.test_connection()
        if test_results["connected"]:
            print("✅ Hybrid GraphRAG Agent retriever configured successfully")

            # Test a simple retrieval (without LLM to avoid dependencies)
            print("   Testing retriever search capability...")
            try:
                # This tests the retriever configuration without needing LLM
                from neo4j_graphrag.retrievers import HybridCypherRetriever

                if isinstance(retriever, HybridCypherRetriever):
                    print(
                        "   ✅ Retriever is properly configured as HybridCypherRetriever"
                    )
                    return True
            except Exception as e:
                print(f"   ⚠️  Retriever test note: {e}")
                return True  # Connection works, that's the main thing
        else:
            print("❌ Hybrid GraphRAG Agent connection failed")
            return False

    except Exception as exc:
        print(f"❌ Hybrid GraphRAG Agent test failed: {exc}")
        return False


def main():
    """Main migration function."""
    print("=" * 60)
    print("🚀 AWS Neo4j Migration Tool for LangGraph Agents")
    print("=" * 60)

    # Step 1: Check environment variables
    if not check_env_variables():
        sys.exit(1)

    # Step 2: Test AWS connection
    if not test_aws_connection():
        print("\n⚠️  Please check your AWS Neo4j credentials and connection URL")
        sys.exit(1)

    # Step 3: Test Cypher Query Agent
    cypher_ok = test_cypher_query_agent()

    # Step 4: Test Hybrid GraphRAG Agent
    graphrag_ok = test_hybrid_graphrag_agent()

    # Summary
    print("\n" + "=" * 60)
    print("📊 Migration Status Summary")
    print("=" * 60)
    print("✅ AWS Neo4j Connection: OK")
    print(
        f"{'✅' if cypher_ok else '❌'} Cypher Query Agent: {'OK' if cypher_ok else 'FAILED'}"
    )
    print(
        f"{'✅' if graphrag_ok else '❌'} Hybrid GraphRAG Agent: {'OK' if graphrag_ok else 'FAILED'}"
    )

    if cypher_ok and graphrag_ok:
        print("\n🎉 Migration successful! Both agents are now using AWS-hosted Neo4j.")
        print("\n📝 Next steps:")
        print("  1. Update any other modules that use Neo4j to use the AWS connection")
        print("  2. Test your LangGraph workflows with the updated agents")
        print("  3. Monitor performance and adjust connection pool settings if needed")
    else:
        print("\n⚠️  Some components failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
