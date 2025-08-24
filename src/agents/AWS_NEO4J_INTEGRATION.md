# AWS Neo4j Integration for LangGraph Agents

## Overview

This document describes the integration of AWS-hosted Neo4j database with the cypher_query_agent and hybrid_graphRAG_agent in the LangGraph workflow.

## Changes Made

### 1. Cypher Query Agent (`cypher_query_agent/`)

#### New Files:
- **`cypher_runner.py`**: New module that replaces the old cypher runner with AWS Neo4j connection
  - Uses the `Neo4jConnection` class from `KnowledgeGraphDB.neo4j_aws_hosted_db.connection`
  - Provides `run_cypher()` and `safe_run_cypher()` functions with retry logic
  - Includes connection testing functionality

#### Modified Files:
- **`agent_logic.py`**: Updated import to use the new `safe_run_cypher` from local cypher_runner module

### 2. Hybrid GraphRAG Agent (`hybrid_graphRAG_agent/`)

#### New Files:
- **`retriever_aws.py`**: New retriever module configured for AWS Neo4j
  - Uses AWS-hosted Neo4j connection via `get_connection()`
  - Configures vector and full-text indexes
  - Maintains the same retrieval query structure for compatibility

#### Modified Files:
- **`knowledge_graph_search.py`**: Updated import to use `retriever_aws` instead of the old retriever

### 3. Configuration Files

#### New Files:
- **`env.example`**: Template for environment variables including:
  - AWS Neo4j connection settings
  - Legacy variable mappings for backward compatibility
  - Cohere API key for embeddings
  - LLM configuration (Azure/Bedrock)

- **`migrate_to_aws_neo4j.py`**: Migration and testing script that:
  - Checks required environment variables
  - Tests AWS Neo4j connection
  - Validates both agents are working correctly

## Setup Instructions

### 1. Environment Configuration

Copy the example environment file and configure your credentials:

```bash
cp env.example .env
```

Edit `.env` and set the following required variables:

```env
# AWS Neo4j Database
AWS_NEO4J_CONNECTION_URL=http://44.243.196.65:7474/
AWS_NEO4J_USERNAME=neo4j
AWS_NEO4J_PASSWORD=your_actual_password

# For backward compatibility
NEO4J_CONNECTION_URI_UPGRADED=bolt://44.243.196.65:7687
NEO4J_USERNAME_UPGRADED=neo4j
NEO4J_PASSWORD_UPGRADED=your_actual_password

# Embeddings
COHERE_API_KEY=your_cohere_api_key
```

### 2. Test the Migration

Run the migration test script to verify everything is working:

```bash
uv run src/agents/migrate_to_aws_neo4j.py
```

This will:
- Check environment variables
- Test AWS Neo4j connection
- Validate cypher_query_agent functionality
- Validate hybrid_graphRAG_agent functionality

### 3. Test Individual Agents

#### Test Cypher Query Agent:
```bash
uv run -m src.agents.cypher_query_agent.cypher_runner
```

#### Test Hybrid GraphRAG Agent:
```bash
uv run -m src.agents.hybrid_graphRAG_agent.retriever_aws
```

## Usage in LangGraph Workflows

Both agents now automatically use the AWS-hosted Neo4j database. No changes are required to the graph builders or high-level workflow code.

### Example: Running Cypher Query Agent

```python
from src.agents.cypher_query_agent.graph_builder import graph

async def run_query():
    async for chunk in graph.astream(
        {"question": "List all projects in the database"},
        stream_mode="updates",
        subgraphs=True,
    ):
        # Process results
        pass
```

### Example: Running Hybrid GraphRAG Agent

```python
from src.agents.hybrid_graphRAG_agent.graph_builder import graph

async def run_graphrag():
    async for chunk in graph.astream(
        {"question": "What information do you have about biosólidos projects?"},
        stream_mode="updates",
        subgraphs=True,
    ):
        # Process results
        pass
```

## Key Features

### Connection Management
- **Connection Pooling**: Efficient connection reuse with configurable pool size
- **Retry Logic**: Automatic retry on transient errors
- **Lazy Initialization**: Connections are created only when needed
- **Singleton Pattern**: Single connection instance shared across the application

### Error Handling
- Graceful error handling with informative error messages
- Automatic reconnection on connection loss
- JSON error responses for failed queries

### Performance Optimizations
- Connection pool configuration (max 50 connections by default)
- Configurable connection lifetime (3600 seconds default)
- Efficient query execution with parameterized queries

## Troubleshooting

### Connection Issues

If you encounter connection problems:

1. Verify Neo4j is accessible:
   ```bash
   curl http://44.243.196.65:7474/
   ```

2. Check credentials are correct in `.env`

3. Test connection directly:
   ```python
   from KnowledgeGraphDB.neo4j_aws_hosted_db.connection import get_connection
   conn = get_connection()
   print(conn.test_connection())
   ```

### Index Creation Errors

The system will automatically create required indexes. If you see index-related warnings, they can usually be ignored if the indexes already exist.

### Performance Issues

Adjust connection pool settings in your `.env`:

```env
NEO4J_MAX_CONNECTION_POOL_SIZE=100  # Increase for high concurrency
NEO4J_CONNECTION_ACQUISITION_TIMEOUT=120.0  # Increase timeout
```

## Backward Compatibility

The integration maintains backward compatibility by:
1. Supporting legacy environment variable names
2. Keeping the same function signatures for `run_cypher()` and `safe_run_cypher()`
3. Maintaining the same retriever interface for GraphRAG

## Future Improvements

Potential enhancements:
1. Add connection health monitoring
2. Implement query caching for frequently used queries
3. Add metrics collection for query performance
4. Support for read replicas for scaling
