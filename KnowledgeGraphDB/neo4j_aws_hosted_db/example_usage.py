#!/usr/bin/env python3
"""Example usage of the Neo4j connection module.

This script demonstrates various features of the connection module
including basic queries, transactions, and GraphRAG patterns.
"""

import logging

from connection import Neo4jConnection, close_default_connection, get_connection


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_basic_connection():
    """Demonstrate basic connection and query execution."""
    logger.info("=" * 50)
    logger.info("Example 1: Basic Connection and Query")
    logger.info("=" * 50)

    # Create connection
    conn = Neo4jConnection()

    try:
        # Execute a simple query
        records, summary, keys = conn.execute_query(
            "RETURN 'Hello from Neo4j!' AS greeting, datetime() AS timestamp"
        )

        for record in records:
            logger.info(f"Greeting: {record['greeting']}")
            logger.info(f"Timestamp: {record['timestamp']}")

        # Get database info
        diagnostics = conn.test_connection()
        logger.info(f"Connected to Neo4j version: {diagnostics['server_version']}")

    finally:
        conn.close()
        logger.info("Connection closed\n")


def example_context_manager():
    """Demonstrate using context manager for automatic cleanup."""
    logger.info("=" * 50)
    logger.info("Example 2: Context Manager")
    logger.info("=" * 50)

    with Neo4jConnection() as conn:
        # Create and query nodes
        conn.execute_query(
            """
            CREATE (n:Example {name: 'Context Manager Test', created: datetime()})
            RETURN n
            """
        )

        records, _, _ = conn.execute_query("MATCH (n:Example) RETURN count(n) AS count")
        logger.info(f"Created nodes: {records[0]['count']}")

        # Clean up
        conn.execute_query("MATCH (n:Example) DELETE n")

    logger.info("Connection automatically closed\n")


def example_parameterized_queries():
    """Demonstrate safe parameterized queries."""
    logger.info("=" * 50)
    logger.info("Example 3: Parameterized Queries")
    logger.info("=" * 50)

    conn = get_connection()

    # Create users with parameters
    users = [
        {"name": "Alice", "age": 30, "city": "New York"},
        {"name": "Bob", "age": 25, "city": "San Francisco"},
        {"name": "Charlie", "age": 35, "city": "London"},
    ]

    for user in users:
        conn.execute_query(
            "CREATE (u:User {name: $name, age: $age, city: $city})", user
        )
        logger.info(f"Created user: {user['name']}")

    # Query with parameters
    records, _, _ = conn.execute_query(
        "MATCH (u:User) WHERE u.age > $min_age RETURN u.name AS name, u.age AS age ORDER BY age",
        {"min_age": 26},
    )

    logger.info("Users older than 26:")
    for record in records:
        logger.info(f"  - {record['name']}: {record['age']} years old")

    # Clean up
    conn.execute_query("MATCH (n:User) DELETE n")
    logger.info("")


def example_transactions():
    """Demonstrate transaction usage."""
    logger.info("=" * 50)
    logger.info("Example 4: Transactions")
    logger.info("=" * 50)

    conn = get_connection()

    # Write transaction function
    def create_company_structure(tx):
        # Create company and departments in a single transaction
        result = tx.run("""
            CREATE (company:Company {name: 'TechCorp', founded: 2020})
            CREATE (eng:Department {name: 'Engineering'})
            CREATE (sales:Department {name: 'Sales'})
            CREATE (hr:Department {name: 'HR'})

            CREATE (company)-[:HAS_DEPARTMENT]->(eng)
            CREATE (company)-[:HAS_DEPARTMENT]->(sales)
            CREATE (company)-[:HAS_DEPARTMENT]->(hr)

            RETURN company.name AS company, count{(company)-[:HAS_DEPARTMENT]->()} AS departments
        """)
        return result.single()

    # Execute write transaction
    result = conn.execute_write_transaction(create_company_structure)
    logger.info(f"Created {result['company']} with {result['departments']} departments")

    # Read transaction function
    def get_company_info(tx):
        result = tx.run("""
            MATCH (c:Company)-[:HAS_DEPARTMENT]->(d:Department)
            RETURN c.name AS company, collect(d.name) AS departments
        """)
        return result.single()

    # Execute read transaction
    info = conn.execute_read_transaction(get_company_info)
    logger.info(f"Company: {info['company']}")
    logger.info(f"Departments: {', '.join(info['departments'])}")

    # Clean up
    conn.execute_query("MATCH (n) WHERE n:Company OR n:Department DETACH DELETE n")
    logger.info("")


def example_graphrag_pattern():
    """Demonstrate GraphRAG document and entity patterns."""
    logger.info("=" * 50)
    logger.info("Example 5: GraphRAG Pattern")
    logger.info("=" * 50)

    conn = get_connection()

    # Create document with chunks and extracted entities
    query = """
    // Create document
    CREATE (doc:Document {
        id: 'doc_example',
        title: 'Quarterly Report',
        date: date('2024-01-15')
    })

    // Create text chunks
    CREATE (chunk1:Chunk {
        id: 'chunk_1',
        text: 'Sarah Johnson, CEO of InnovateTech, announced record profits.',
        position: 1
    })

    CREATE (chunk2:Chunk {
        id: 'chunk_2',
        text: 'The company, based in Austin, Texas, grew by 45% this quarter.',
        position: 2
    })

    // Link chunks to document
    CREATE (doc)-[:HAS_CHUNK]->(chunk1)
    CREATE (doc)-[:HAS_CHUNK]->(chunk2)
    CREATE (chunk1)-[:NEXT_CHUNK]->(chunk2)

    // Extract and create entities
    CREATE (sarah:Person {name: 'Sarah Johnson', role: 'CEO'})
    CREATE (company:Organization {name: 'InnovateTech'})
    CREATE (austin:Location {name: 'Austin', state: 'Texas'})

    // Create semantic relationships
    CREATE (sarah)-[:LEADS]->(company)
    CREATE (company)-[:LOCATED_IN]->(austin)

    // Link entities to source chunks
    CREATE (sarah)-[:MENTIONED_IN]->(chunk1)
    CREATE (company)-[:MENTIONED_IN]->(chunk1)
    CREATE (company)-[:MENTIONED_IN]->(chunk2)
    CREATE (austin)-[:MENTIONED_IN]->(chunk2)

    RETURN doc, chunk1, chunk2, sarah, company, austin
    """

    conn.execute_query(query)
    logger.info("Created GraphRAG structure")

    # Query the knowledge graph
    kg_query = """
    MATCH (p:Person)-[:LEADS]->(org:Organization)-[:LOCATED_IN]->(loc:Location)
    MATCH (p)-[:MENTIONED_IN]->(chunk:Chunk)<-[:HAS_CHUNK]-(doc:Document)
    RETURN
        p.name AS person,
        p.role AS role,
        org.name AS organization,
        loc.name AS location,
        doc.title AS document,
        collect(DISTINCT chunk.text) AS source_texts
    """

    records, _, _ = conn.execute_query(kg_query)

    for record in records:
        logger.info("\nExtracted Information:")
        logger.info(f"  Person: {record['person']} ({record['role']})")
        logger.info(f"  Organization: {record['organization']}")
        logger.info(f"  Location: {record['location']}")
        logger.info(f"  Document: {record['document']}")
        logger.info("  Source texts:")
        for text in record["source_texts"]:
            logger.info(f"    - {text}")

    # Clean up
    conn.execute_query("""
        MATCH (n)
        WHERE n:Document OR n:Chunk OR n:Person OR n:Organization OR n:Location
        DETACH DELETE n
    """)
    logger.info("")


def example_batch_operations():
    """Demonstrate batch insert operations."""
    logger.info("=" * 50)
    logger.info("Example 6: Batch Operations")
    logger.info("=" * 50)

    conn = get_connection()

    # Prepare batch data
    products = [
        {
            "id": f"prod_{i}",
            "name": f"Product {i}",
            "price": 10.0 * i,
            "category": f"Category {i % 3}",
        }
        for i in range(1, 21)
    ]

    # Batch insert
    records, _, _ = conn.execute_query(
        """
        UNWIND $products AS product
        CREATE (p:Product {
            id: product.id,
            name: product.name,
            price: product.price,
            category: product.category
        })
        RETURN count(p) AS created_count
        """,
        {"products": products},
    )

    logger.info(f"Batch created {records[0]['created_count']} products")

    # Aggregate query
    records, _, _ = conn.execute_query(
        """
        MATCH (p:Product)
        RETURN
            p.category AS category,
            count(p) AS count,
            avg(p.price) AS avg_price
        ORDER BY category
        """
    )

    logger.info("Product statistics by category:")
    for record in records:
        logger.info(
            f"  {record['category']}: {record['count']} products, "
            f"avg price: ${record['avg_price']:.2f}"
        )

    # Clean up
    conn.execute_query("MATCH (n:Product) DELETE n")
    logger.info("")


def main():
    """Run all examples."""
    logger.info("\n" + "=" * 70)
    logger.info("NEO4J CONNECTION MODULE EXAMPLES")
    logger.info("=" * 70 + "\n")

    try:
        # Run examples
        example_basic_connection()
        example_context_manager()
        example_parameterized_queries()
        example_transactions()
        example_graphrag_pattern()
        example_batch_operations()

        logger.info("=" * 70)
        logger.info("All examples completed successfully!")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise
    finally:
        # Ensure default connection is closed
        close_default_connection()


if __name__ == "__main__":
    main()
