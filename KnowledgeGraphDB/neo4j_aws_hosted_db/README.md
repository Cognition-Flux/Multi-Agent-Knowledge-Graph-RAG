# Neo4j AWS Hosted Database Connection Module

A robust Python connection module for AWS-hosted Neo4j databases with comprehensive error handling, connection pooling, retry logic, and extensive testing.

## Features

- **Robust Connection Management**: Automatic Bolt protocol conversion from HTTP URLs
- **Connection Pooling**: Efficient connection reuse with configurable pool size
- **Retry Logic**: Automatic retry on transient errors with exponential backoff
- **Two-Tier Verification**: Ensures both connectivity and database accessibility
- **Context Managers**: Clean resource management with automatic cleanup
- **Transaction Support**: Both read and write transactions with retry logic
- **Comprehensive Testing**: Full test suite covering all functionality
- **GraphRAG Ready**: Built-in support for document/chunk structures and entity extraction patterns

## Installation

1. Install required dependencies:
```bash
uv add neo4j python-dotenv pytest
```

2. Copy the example environment file and configure your credentials:
```bash
cp env.example .env
# Edit .env with your actual credentials
```

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Required
AWS_NEO4J_CONNECTION_URL=http://44.243.196.65:7474/
AWS_NEO4J_USERNAME=neo4j
AWS_NEO4J_PASSWORD=your_password_here

# Optional
AWS_NEO4J_DATABASE=neo4j  # Target database (default: "neo4j")
```

The connection module automatically converts HTTP URLs (port 7474) to Bolt protocol (port 7687).

## Usage

### Basic Connection

```python
from connection import Neo4jConnection

# Create connection instance
conn = Neo4jConnection()

# Execute a simple query
records, summary, keys = conn.execute_query("RETURN 1 AS number")
print(records[0]["number"])  # Output: 1

# Close connection when done
conn.close()
```

### Using Context Manager

```python
from connection import Neo4jConnection

# Automatic connection management
with Neo4jConnection() as conn:
    records, _, _ = conn.execute_query("MATCH (n) RETURN count(n) AS count")
    print(f"Total nodes: {records[0]['count']}")
# Connection automatically closed
```

### Singleton Pattern

```python
from connection import get_connection, close_default_connection

# Get or create default connection
conn = get_connection()

# Use the connection
records, _, _ = conn.execute_query("RETURN 'Hello, Neo4j!' AS message")
print(records[0]["message"])

# Close default connection when done
close_default_connection()
```

### Parameterized Queries

```python
conn = get_connection()

# Safe parameterized query
parameters = {
    "name": "Alice",
    "age": 30,
    "city": "San Francisco"
}

query = """
CREATE (p:Person {name: $name, age: $age, city: $city})
RETURN p
"""

records, _, _ = conn.execute_query(query, parameters)
person = records[0]["p"]
print(f"Created person: {person['name']}, age {person['age']}")
```

### Transactions

```python
conn = get_connection()

# Write transaction
def create_nodes(tx, count):
    result = tx.run(
        "UNWIND range(1, $count) AS id CREATE (n:Node {id: id}) RETURN count(n) AS created",
        count=count
    )
    return result.single()["created"]

nodes_created = conn.execute_write_transaction(create_nodes, 10)
print(f"Created {nodes_created} nodes")

# Read transaction
def count_nodes(tx):
    result = tx.run("MATCH (n:Node) RETURN count(n) AS total")
    return result.single()["total"]

total = conn.execute_read_transaction(count_nodes)
print(f"Total nodes: {total}")
```

### Session Management

```python
conn = get_connection()

# Use session context manager
with conn.session() as session:
    # Multiple operations in same session
    session.run("CREATE (n:Product {name: 'GraphDB', version: '5.0'})")
    session.run("CREATE (n:Product {name: 'Neo4j', version: '5.13'})")

    result = session.run("MATCH (p:Product) RETURN p.name AS name ORDER BY name")
    products = [record["name"] for record in result]
    print(f"Products: {products}")
```

### Connection Diagnostics

```python
conn = get_connection()

# Get detailed connection diagnostics
diagnostics = conn.test_connection()

print(f"Connected: {diagnostics['connected']}")
print(f"Server Version: {diagnostics['server_version']}")
print(f"Database: {diagnostics['database']}")
print(f"Database Exists: {diagnostics['database_exists']}")

if diagnostics.get('database_info'):
    print(f"Database Name: {diagnostics['database_info']['name']}")
    print(f"Creation Date: {diagnostics['database_info']['creation_date']}")
```

### GraphRAG Pattern Example

```python
conn = get_connection()

# Create document and chunk structure
query = """
// Create document
CREATE (doc:Document {
    id: 'doc_001',
    title: 'Sample Document',
    source: 'example.pdf'
})

// Create chunks
CREATE (chunk1:Chunk {
    id: 'chunk_001',
    text: 'Alice is the CEO of TechCorp.',
    embedding: [0.1, 0.2, 0.3]  // Vector embedding
})

CREATE (chunk2:Chunk {
    id: 'chunk_002',
    text: 'TechCorp is based in Silicon Valley.',
    embedding: [0.4, 0.5, 0.6]
})

// Link chunks to document
CREATE (doc)-[:HAS_CHUNK]->(chunk1)
CREATE (doc)-[:HAS_CHUNK]->(chunk2)
CREATE (chunk1)-[:NEXT_CHUNK]->(chunk2)

// Extract entities
CREATE (alice:Person {name: 'Alice', role: 'CEO'})
CREATE (techcorp:Company {name: 'TechCorp'})
CREATE (location:Location {name: 'Silicon Valley'})

// Create relationships
CREATE (alice)-[:WORKS_AT {position: 'CEO'}]->(techcorp)
CREATE (techcorp)-[:LOCATED_IN]->(location)

// Link entities to source chunks
CREATE (alice)-[:EXTRACTED_FROM]->(chunk1)
CREATE (techcorp)-[:EXTRACTED_FROM]->(chunk1)
CREATE (techcorp)-[:EXTRACTED_FROM]->(chunk2)
CREATE (location)-[:EXTRACTED_FROM]->(chunk2)

RETURN doc, chunk1, chunk2, alice, techcorp, location
"""

records, _, _ = conn.execute_query(query)
print("GraphRAG structure created successfully")

# Query the knowledge graph
kg_query = """
MATCH (p:Person)-[r:WORKS_AT]->(c:Company)-[:LOCATED_IN]->(l:Location)
MATCH (p)-[:EXTRACTED_FROM]->(chunk:Chunk)
RETURN p.name AS person, r.position AS position,
       c.name AS company, l.name AS location,
       collect(DISTINCT chunk.text) AS source_texts
"""

records, _, _ = conn.execute_query(kg_query)
for record in records:
    print(f"{record['person']} is {record['position']} at {record['company']} in {record['location']}")
    print(f"Sources: {record['source_texts']}")
```

## Running Tests

The module includes a comprehensive test suite covering all functionality:

```bash
# Run all tests
uv run python test_connection.py

# Run with pytest directly for more options
uv run pytest test_connection.py -v

# Run specific test class
uv run pytest test_connection.py::TestNeo4jConnection -v

# Run specific test
uv run pytest test_connection.py::TestNeo4jConnection::test_connection_initialization -v

# Run with coverage
uv run pytest test_connection.py --cov=connection --cov-report=html
```

### Test Coverage

The test suite includes:

- **Connection Tests**: Initialization, connectivity verification, custom parameters
- **Query Execution**: Simple queries, parameterized queries, large result sets
- **Transactions**: Write and read transactions with retry logic
- **Session Management**: Context managers and session operations
- **Error Handling**: Authentication errors, invalid queries, transient errors
- **Batch Operations**: Bulk inserts and updates
- **Concurrent Operations**: Thread-safe connection pooling
- **GraphRAG Patterns**: Document/chunk structures, entity extraction
- **Diagnostics**: Connection health checks and server information

## Error Handling

The module provides comprehensive error handling with specific exceptions:

```python
from neo4j.exceptions import AuthError, ServiceUnavailable, ClientError

try:
    conn = Neo4jConnection()
    records, _, _ = conn.execute_query("MATCH (n) RETURN n")
except AuthError:
    print("Authentication failed. Check credentials.")
except ServiceUnavailable:
    print("Neo4j service is not available.")
except ClientError as e:
    print(f"Query error: {e}")
```

## Retry Logic

Automatic retry on transient errors with configurable parameters:

```python
# Configure retry behavior
conn = Neo4jConnection(
    max_retry_attempts=5,  # Try up to 5 times
    retry_delay=2.0        # Wait 2 seconds between attempts
)

# Operations automatically retry on transient errors
records, _, _ = conn.execute_query("MATCH (n) RETURN n")
```

## Connection Pooling

Efficient connection reuse with configurable pool settings:

```python
# Configure connection pool
conn = Neo4jConnection(
    max_connection_pool_size=100,      # Maximum 100 connections
    max_connection_lifetime=1800,      # Connections live for 30 minutes
    connection_acquisition_timeout=30  # Wait up to 30 seconds for connection
)
```

## Best Practices

1. **Use Environment Variables**: Store credentials in `.env` file, never commit them
2. **Use Context Managers**: Ensures proper resource cleanup
3. **Parameterize Queries**: Prevent Cypher injection attacks
4. **Handle Errors**: Always wrap operations in try-except blocks
5. **Use Transactions**: Group related operations in transactions
6. **Monitor Connections**: Use `test_connection()` for health checks
7. **Close Connections**: Always close connections when done

## Troubleshooting

### Connection Issues

1. **Verify Neo4j is running**: Check if the database is accessible
2. **Check credentials**: Ensure username/password are correct
3. **Verify network**: Ensure the host is reachable from your machine
4. **Check ports**: Bolt uses port 7687, not 7474 (HTTP)

### Query Issues

1. **Syntax errors**: Validate Cypher syntax in Neo4j Browser first
2. **Missing nodes/relationships**: Check if data exists
3. **Performance**: Create indexes for frequently queried properties

### Test Failures

1. **Clean database**: Tests expect a clean database state
2. **Dependencies**: Ensure all required packages are installed
3. **Environment**: Check `.env` file configuration

## License

This module is part of the CSW-NVIRO project.
