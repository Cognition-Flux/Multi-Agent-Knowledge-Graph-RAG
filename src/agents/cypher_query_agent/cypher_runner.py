"""Cypher runner module for the Cypher Query Agent using AWS-hosted Neo4j.

This module provides the interface to execute Cypher queries against the
AWS-hosted Neo4j database, replacing the previous local database connection.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from KnowledgeGraphDB.neo4j_aws_hosted_db.connection import (
    get_connection,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_cypher(
    query: str,
    parameters: dict[str, Any] | None = None,
    *,
    close_after: bool = False,
) -> list[dict[str, Any]]:
    """Run a Cypher query against the AWS-hosted Neo4j database.

    Parameters
    ----------
    query : str
        The Cypher query string to execute.
    parameters : dict[str, Any] | None
        Optional dictionary of parameters for the query.
    close_after : bool
        If True, close the connection after executing the query.
        Default is False to keep the connection alive for multiple queries.

    Returns:
    -------
    list[dict[str, Any]]
        List of result records as dictionaries.

    Raises:
    ------
    Exception
        If the query execution fails.
    """
    if parameters is None:
        parameters = {}

    try:
        # Get the singleton connection instance
        connection = get_connection()

        # Execute the query with retry logic
        records, summary, keys = connection.execute_query(query, parameters)

        # Convert records to list of dictionaries
        result = [dict(record) for record in records]

        # Close connection if requested
        if close_after:
            connection.close()

        logger.debug(f"Query executed successfully: {query[:100]}...")
        return result

    except Exception as exc:
        logger.error(f"Error executing Cypher query: {exc}")
        logger.error(f"Query: {query}")
        raise


def safe_run_cypher(query: str) -> str:
    """Run Cypher query and return results as a JSON string.

    This is a wrapper function that handles errors gracefully,
    returning error information as JSON if the query fails.

    Parameters
    ----------
    query : str
        The Cypher query to execute.

    Returns:
    -------
    str
        JSON string containing either the query results or error information.
    """
    try:
        result = run_cypher(query)
        # Ensure results are JSON serializable
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        logger.error(f"Error in safe_run_cypher: {exc}")
        return json.dumps({"error": f"ERROR: {exc}"}, ensure_ascii=False)


def test_connection() -> bool:
    """Test the connection to the AWS-hosted Neo4j database.

    Returns:
    -------
    bool
        True if the connection is successful, False otherwise.
    """
    try:
        connection = get_connection()
        test_results = connection.test_connection()

        if test_results["connected"]:
            logger.info(f"✅ Connected to Neo4j at {test_results['uri']}")
            logger.info(f"   Database: {test_results['database']}")
            if test_results.get("server_version"):
                logger.info(f"   Server version: {test_results['server_version']}")
            return True
        else:
            logger.error(
                f"❌ Failed to connect: {test_results.get('error', 'Unknown error')}"
            )
            return False

    except Exception as exc:
        logger.error(f"❌ Connection test failed: {exc}")
        return False


if __name__ == "__main__":
    # Test the connection and run a sample query
    print("Testing AWS-hosted Neo4j connection...")

    if test_connection():
        print("\n📊 Running sample query...")
        try:
            # Simple test query
            result = run_cypher("MATCH (n) RETURN count(n) as node_count LIMIT 1")
            print(f"✅ Query successful! Node count: {result}")

            # Test with parameters
            result_with_params = run_cypher(
                "RETURN $test_param as test_value",
                {"test_param": "Hello from AWS Neo4j!"},
            )
            print(f"✅ Parameterized query result: {result_with_params}")

        except Exception as e:
            print(f"❌ Query failed: {e}")
    else:
        print("❌ Could not establish connection to AWS Neo4j")
