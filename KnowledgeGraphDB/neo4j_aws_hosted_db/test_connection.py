"""Comprehensive test suite for Neo4j AWS hosted database connection.

This module provides thorough testing of the Neo4j connection,
including connectivity, CRUD operations, transactions, and error handling.

uv run python KnowledgeGraphDB/neo4j_aws_hosted_db/test_connection.py

"""

import logging

import pytest
from connection import Neo4jConnection, close_default_connection, get_connection
from dotenv import load_dotenv
from neo4j.exceptions import AuthError, ClientError, TransientError


# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)


class TestNeo4jConnection:
    """Test suite for Neo4j connection functionality."""

    @classmethod
    def setup_class(cls):
        """Setup test class - runs once before all tests."""
        cls.test_node_label = "TestNode"
        cls.test_relationship_type = "TEST_RELATES_TO"
        logger.info("Starting Neo4j connection tests")

    @classmethod
    def teardown_class(cls):
        """Teardown test class - cleanup after all tests."""
        # Clean up any test data
        try:
            conn = get_connection()
            conn.execute_query(f"MATCH (n:{cls.test_node_label}) DETACH DELETE n")
            close_default_connection()
            logger.info("Test cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup error (non-critical): {e}")

    def setup_method(self, method):
        """Setup for each test method."""
        logger.info(f"Running test: {method.__name__}")
        # Clean up any existing test nodes before each test
        try:
            conn = get_connection()
            conn.execute_query(f"MATCH (n:{self.test_node_label}) DETACH DELETE n")
        except Exception:
            pass

    def test_connection_initialization(self):
        """Test basic connection initialization."""
        conn = Neo4jConnection()

        assert conn.uri.startswith("bolt://")
        assert conn.username == "neo4j"
        assert conn.password is not None
        assert conn.database == "neo4j"

        # Test that driver is not created until accessed
        assert conn._driver is None

        # Access driver to trigger lazy initialization
        driver = conn.driver
        assert driver is not None
        assert conn._driver is not None

        conn.close()

    def test_connection_with_custom_parameters(self):
        """Test connection with custom configuration parameters."""
        conn = Neo4jConnection(
            max_connection_pool_size=10,
            connection_acquisition_timeout=30.0,
            max_retry_attempts=5,
            retry_delay=2.0,
        )

        assert conn.max_connection_pool_size == 10
        assert conn.connection_acquisition_timeout == 30.0
        assert conn.max_retry_attempts == 5
        assert conn.retry_delay == 2.0

        conn.close()

    def test_connectivity_verification(self):
        """Test two-tier connectivity verification."""
        conn = Neo4jConnection()

        # Should successfully connect and verify
        driver = conn.driver
        assert driver is not None

        # Test is_connected method
        assert conn.is_connected() is True

        # Close and verify disconnection
        conn.close()
        assert conn.is_connected() is False

    def test_execute_query(self):
        """Test basic query execution."""
        conn = get_connection()

        # Test simple query
        records, summary, keys = conn.execute_query(
            "RETURN 1 AS number, 'test' AS text"
        )

        assert len(records) == 1
        assert records[0]["number"] == 1
        assert records[0]["text"] == "test"
        assert "number" in keys
        assert "text" in keys

    def test_parameterized_query(self):
        """Test query execution with parameters."""
        conn = get_connection()

        parameters = {"name": "Test User", "age": 30, "active": True}

        query = """
        CREATE (n:TestNode {name: $name, age: $age, active: $active})
        RETURN n
        """

        records, _, _ = conn.execute_query(query, parameters)

        assert len(records) == 1
        node = records[0]["n"]
        assert node["name"] == "Test User"
        assert node["age"] == 30
        assert node["active"] is True

    def test_write_transaction(self):
        """Test write transaction execution."""
        conn = get_connection()

        def create_test_nodes(tx, count):
            result = tx.run(
                """
                UNWIND range(1, $count) AS id
                CREATE (n:TestNode {id: id, created_at: datetime()})
                RETURN count(n) AS nodes_created
                """,
                count=count,
            )
            return result.single()["nodes_created"]

        nodes_created = conn.execute_write_transaction(create_test_nodes, 5)
        assert nodes_created == 5

        # Verify nodes were created
        records, _, _ = conn.execute_query(
            "MATCH (n:TestNode) RETURN count(n) AS count"
        )
        assert records[0]["count"] == 5

    def test_read_transaction(self):
        """Test read transaction execution."""
        conn = get_connection()

        # First create some test data
        conn.execute_query(
            """
            CREATE (n1:TestNode {name: 'Node1'})
            CREATE (n2:TestNode {name: 'Node2'})
            CREATE (n3:TestNode {name: 'Node3'})
            """
        )

        def count_nodes(tx):
            result = tx.run("MATCH (n:TestNode) RETURN count(n) AS count")
            return result.single()["count"]

        count = conn.execute_read_transaction(count_nodes)
        assert count == 3

    def test_session_context_manager(self):
        """Test session context manager."""
        conn = get_connection()

        with conn.session() as session:
            # Create a node
            result = session.run(
                "CREATE (n:TestNode {name: $name}) RETURN n", name="Session Test"
            )
            node = result.single()["n"]
            assert node["name"] == "Session Test"

        # Verify node exists after session closes
        records, _, _ = conn.execute_query(
            "MATCH (n:TestNode {name: 'Session Test'}) RETURN n"
        )
        assert len(records) == 1

    def test_connection_context_manager(self):
        """Test connection context manager."""
        with Neo4jConnection() as conn:
            assert conn.is_connected() is True

            records, _, _ = conn.execute_query("RETURN 'connected' AS status")
            assert records[0]["status"] == "connected"

        # Connection should be closed after exiting context
        assert conn.is_connected() is False

    def test_complex_graph_operations(self):
        """Test complex graph operations with nodes and relationships."""
        conn = get_connection()

        # Create a small graph
        query = """
        CREATE (alice:TestNode:Person {name: 'Alice', age: 30})
        CREATE (bob:TestNode:Person {name: 'Bob', age: 35})
        CREATE (charlie:TestNode:Person {name: 'Charlie', age: 25})
        CREATE (project:TestNode:Project {name: 'GraphRAG', status: 'active'})

        CREATE (alice)-[:WORKS_ON {role: 'Lead'}]->(project)
        CREATE (bob)-[:WORKS_ON {role: 'Developer'}]->(project)
        CREATE (charlie)-[:WORKS_ON {role: 'Tester'}]->(project)

        CREATE (alice)-[:KNOWS {since: 2020}]->(bob)
        CREATE (bob)-[:KNOWS {since: 2021}]->(charlie)

        RETURN alice, bob, charlie, project
        """

        records, _, _ = conn.execute_query(query)
        assert len(records) == 1

        # Query the graph structure
        path_query = """
        MATCH path = (alice:TestNode:Person {name: 'Alice'})-[:KNOWS*1..2]-(other:Person)
        RETURN alice.name AS start, collect(other.name) AS connected_people
        """

        records, _, _ = conn.execute_query(path_query)
        assert len(records) > 0
        assert "Bob" in records[0]["connected_people"]

    def test_batch_operations(self):
        """Test batch insert and update operations."""
        conn = get_connection()

        # Batch insert
        batch_data = [
            {"id": i, "name": f"Item {i}", "value": i * 10} for i in range(1, 101)
        ]

        query = """
        UNWIND $batch AS item
        CREATE (n:TestNode:BatchItem {id: item.id, name: item.name, value: item.value})
        RETURN count(n) AS created_count
        """

        records, _, _ = conn.execute_query(query, {"batch": batch_data})
        assert records[0]["created_count"] == 100

        # Batch update
        update_query = """
        MATCH (n:TestNode:BatchItem)
        WHERE n.id % 2 = 0
        SET n.updated = true, n.value = n.value * 2
        RETURN count(n) AS updated_count
        """

        records, _, _ = conn.execute_query(update_query)
        assert records[0]["updated_count"] == 50

    def test_error_handling(self):
        """Test error handling for various scenarios."""
        # Test invalid credentials
        with pytest.raises(AuthError):
            bad_conn = Neo4jConnection(password="wrong_password")
            bad_conn.driver  # Trigger connection attempt

        # Test invalid query syntax
        conn = get_connection()
        with pytest.raises(ClientError):
            conn.execute_query("INVALID CYPHER SYNTAX")

        # Test that querying non-existent labels returns empty results (doesn't raise error)
        records, _, _ = conn.execute_query("MATCH (n:NonExistentLabel) RETURN n")
        assert len(records) == 0  # Should return empty results, not raise error

    def test_retry_logic(self):
        """Test retry logic for transient errors."""
        conn = Neo4jConnection(max_retry_attempts=3, retry_delay=0.1)

        # Create a function that fails twice then succeeds
        attempt_count = {"count": 0}

        @conn.retry_on_transient_error
        def flaky_operation():
            attempt_count["count"] += 1
            if attempt_count["count"] < 3:
                raise TransientError("Simulated transient error")
            return "Success"

        result = flaky_operation()
        assert result == "Success"
        assert attempt_count["count"] == 3

    def test_connection_test_diagnostics(self):
        """Test the connection diagnostics method."""
        conn = get_connection()

        diagnostics = conn.test_connection()

        assert diagnostics["connected"] is True
        assert diagnostics["database_exists"] is True
        assert diagnostics["server_version"] is not None
        assert diagnostics["error"] is None
        assert "database_info" in diagnostics

        logger.info(f"Connection diagnostics: {diagnostics}")

    def test_concurrent_operations(self):
        """Test concurrent database operations."""
        import concurrent.futures

        conn = get_connection()

        def create_node(node_id):
            query = """
            CREATE (n:TestNode:Concurrent {id: $id, created_at: datetime()})
            RETURN n.id AS id
            """
            records, _, _ = conn.execute_query(query, {"id": node_id})
            return records[0]["id"]

        # Execute multiple operations concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_node, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 50
        assert all(isinstance(r, int) for r in results)

        # Verify all nodes were created
        records, _, _ = conn.execute_query(
            "MATCH (n:TestNode:Concurrent) RETURN count(n) AS count"
        )
        assert records[0]["count"] == 50

    def test_database_constraints_and_indexes(self):
        """Test creating and using constraints and indexes."""
        conn = get_connection()

        try:
            # Create a uniqueness constraint
            conn.execute_query(
                "CREATE CONSTRAINT test_unique IF NOT EXISTS "
                "FOR (n:TestNode) REQUIRE n.unique_id IS UNIQUE"
            )

            # Create nodes with unique IDs
            conn.execute_query(
                "CREATE (n:TestNode {unique_id: 'unique_1', data: 'test1'})"
            )

            # Attempt to create duplicate - should fail
            with pytest.raises(ClientError) as exc_info:
                conn.execute_query(
                    "CREATE (n:TestNode {unique_id: 'unique_1', data: 'test2'})"
                )
            assert "constraint" in str(exc_info.value).lower()

        finally:
            # Clean up constraint
            try:
                conn.execute_query("DROP CONSTRAINT test_unique IF EXISTS")
            except Exception:
                pass

    def test_large_result_sets(self):
        """Test handling of large result sets."""
        conn = get_connection()

        # Create a large number of nodes
        # Note: Neo4j doesn't support string multiplication, so we use a fixed string
        large_string = "x" * 100  # Create the string in Python
        conn.execute_query(
            """
            UNWIND range(1, 1000) AS id
            CREATE (n:TestNode:LargeSet {id: id, data: $data})
            """,
            {"data": large_string},
        )

        # Query all nodes
        records, summary, _ = conn.execute_query(
            "MATCH (n:TestNode:LargeSet) RETURN n ORDER BY n.id"
        )

        assert len(records) == 1000
        assert records[0]["n"]["id"] == 1
        assert records[-1]["n"]["id"] == 1000


class TestGraphRAGIntegration:
    """Test suite for GraphRAG-specific functionality."""

    def test_document_and_chunk_structure(self):
        """Test creating document and chunk nodes for GraphRAG."""
        conn = get_connection()

        # Create a document with chunks
        query = """
        CREATE (doc:Document {
            id: 'doc_001',
            title: 'Test Document',
            source: 'test_suite',
            created_at: datetime()
        })

        CREATE (chunk1:Chunk {
            id: 'chunk_001',
            text: 'This is the first chunk of text.',
            embedding: [0.1, 0.2, 0.3],
            position: 0
        })

        CREATE (chunk2:Chunk {
            id: 'chunk_002',
            text: 'This is the second chunk of text.',
            embedding: [0.4, 0.5, 0.6],
            position: 1
        })

        CREATE (doc)-[:HAS_CHUNK]->(chunk1)
        CREATE (doc)-[:HAS_CHUNK]->(chunk2)
        CREATE (chunk1)-[:NEXT_CHUNK]->(chunk2)

        RETURN doc, chunk1, chunk2
        """

        records, _, _ = conn.execute_query(query)
        assert len(records) == 1

        # Query the structure
        verify_query = """
        MATCH (doc:Document {id: 'doc_001'})-[:HAS_CHUNK]->(chunk:Chunk)
        RETURN doc.title AS title, collect(chunk.text) AS chunks
        """

        records, _, _ = conn.execute_query(verify_query)
        assert records[0]["title"] == "Test Document"
        assert len(records[0]["chunks"]) == 2

        # Clean up
        conn.execute_query("MATCH (n:Document|Chunk) DETACH DELETE n")

    def test_entity_extraction_pattern(self):
        """Test entity extraction and linking pattern for GraphRAG."""
        conn = get_connection()

        # Simulate entity extraction from chunks
        query = """
        // Create a chunk
        CREATE (chunk:Chunk {
            id: 'chunk_test',
            text: 'Alice works at TechCorp in San Francisco.'
        })

        // Extract entities
        CREATE (person:Person {name: 'Alice'})
        CREATE (company:Company {name: 'TechCorp'})
        CREATE (location:Location {name: 'San Francisco'})

        // Create relationships
        CREATE (person)-[:WORKS_AT]->(company)
        CREATE (company)-[:LOCATED_IN]->(location)

        // Link entities to source chunk
        CREATE (person)-[:EXTRACTED_FROM]->(chunk)
        CREATE (company)-[:EXTRACTED_FROM]->(chunk)
        CREATE (location)-[:EXTRACTED_FROM]->(chunk)

        RETURN person, company, location, chunk
        """

        records, _, _ = conn.execute_query(query)
        assert len(records) == 1

        # Query the knowledge graph
        kg_query = """
        MATCH (p:Person {name: 'Alice'})-[:WORKS_AT]->(c:Company)
        MATCH (c)-[:LOCATED_IN]->(l:Location)
        MATCH (p)-[:EXTRACTED_FROM]->(chunk:Chunk)
        RETURN p.name AS person, c.name AS company, l.name AS location, chunk.text AS source
        """

        records, _, _ = conn.execute_query(kg_query)
        assert records[0]["person"] == "Alice"
        assert records[0]["company"] == "TechCorp"
        assert records[0]["location"] == "San Francisco"
        assert "Alice works at TechCorp" in records[0]["source"]

        # Clean up
        conn.execute_query(
            "MATCH (n) WHERE n:Person OR n:Company OR n:Location OR n:Chunk DETACH DELETE n"
        )


def run_tests():
    """Run all tests and report results."""
    # Run pytest with verbose output
    exit_code = pytest.main([__file__, "-v", "-s", "--tb=short"])

    if exit_code == 0:
        logger.info("✅ All tests passed successfully!")
    else:
        logger.error("❌ Some tests failed. Check the output above.")

    return exit_code


if __name__ == "__main__":
    # For direct execution
    exit_code = run_tests()
    exit(exit_code)
