"""Connection module for AWS Hosted Neo4j Graph Database.

This module provides a robust connection interface to the Neo4j database
with proper error handling, connection pooling, and retry logic.
"""

import logging
import os
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import (
    AuthError,
    ClientError,
    Neo4jError,
    ServiceUnavailable,
    TransientError,
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(override=True)


class Neo4jConnection:
    """Manages connection to AWS hosted Neo4j database with retry logic and connection pooling."""

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str = "neo4j",
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50,
        connection_acquisition_timeout: float = 60.0,
        max_retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize Neo4j connection manager.

        Args:
            uri: Neo4j connection URI. Defaults to environment variable.
            username: Neo4j username. Defaults to environment variable.
            password: Neo4j password. Defaults to environment variable.
            database: Target database name. Defaults to "neo4j".
            max_connection_lifetime: Maximum connection lifetime in seconds.
            max_connection_pool_size: Maximum size of the connection pool.
            connection_acquisition_timeout: Timeout for acquiring a connection from the pool.
            max_retry_attempts: Maximum number of retry attempts for transient errors.
            retry_delay: Delay between retry attempts in seconds.
        """
        # Connection parameters with fallback to environment variables
        self.uri = uri or self._get_bolt_uri()
        self.username = username or os.getenv("AWS_NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("AWS_NEO4J_PASSWORD")
        self.database = database

        # Connection pool configuration
        self.max_connection_lifetime = max_connection_lifetime
        self.max_connection_pool_size = max_connection_pool_size
        self.connection_acquisition_timeout = connection_acquisition_timeout

        # Retry configuration
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay = retry_delay

        # Driver instance (lazy initialization)
        self._driver: Driver | None = None

        # Validate credentials
        if not self.password:
            raise ValueError(
                "Neo4j password not provided. Set AWS_NEO4J_PASSWORD environment variable."
            )

    def _get_bolt_uri(self) -> str:
        """Convert HTTP URL to Bolt URI if needed."""
        env_url = os.getenv("AWS_NEO4J_CONNECTION_URL", "")

        if env_url.startswith("http://") or env_url.startswith("https://"):
            # Extract host from HTTP URL and convert to Bolt
            host = (
                env_url.replace("http://", "")
                .replace("https://", "")
                .split(":")[0]
                .split("/")[0]
            )
            return f"bolt://{host}:7687"
        elif env_url.startswith("bolt://") or env_url.startswith("neo4j://"):
            return env_url
        else:
            # Default to the provided IP
            return "bolt://44.243.196.65:7687"

    def retry_on_transient_error(self, func):
        """Decorator to retry operations on transient errors."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(self.max_retry_attempts):
                try:
                    return func(*args, **kwargs)
                except (ServiceUnavailable, TransientError) as e:
                    last_error = e
                    if attempt < self.max_retry_attempts - 1:
                        logger.warning(
                            f"Transient error on attempt {attempt + 1}/{self.max_retry_attempts}: {e}. "
                            f"Retrying in {self.retry_delay} seconds..."
                        )
                        time.sleep(
                            self.retry_delay * (attempt + 1)
                        )  # Exponential backoff
                    else:
                        logger.error(f"Max retry attempts reached. Last error: {e}")
                        raise
            raise last_error

        return wrapper

    @property
    def driver(self) -> Driver:
        """Get or create the Neo4j driver instance (lazy initialization)."""
        if self._driver is None:
            self._driver = self._create_driver()
        return self._driver

    def _create_driver(self) -> Driver:
        """Create a new Neo4j driver instance with connection pooling."""
        logger.info(f"Creating Neo4j driver for {self.uri}")

        try:
            driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=self.max_connection_lifetime,
                max_connection_pool_size=self.max_connection_pool_size,
                connection_acquisition_timeout=self.connection_acquisition_timeout,
                # Note: trust parameter removed as it's not available in current driver version
            )

            # Verify connectivity
            self._verify_connectivity(driver)

            return driver

        except AuthError as e:
            logger.error(f"Authentication failed: {e}")
            raise
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable at {self.uri}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating driver: {e}")
            raise

    def _verify_connectivity(self, driver: Driver) -> None:
        """Verify database connectivity with two-tier verification."""
        # Tier 1: Basic connectivity check
        driver.verify_connectivity()
        logger.info("Tier 1 verification passed: Basic connectivity confirmed")

        # Tier 2: Database accessibility check
        try:
            records, summary, keys = driver.execute_query(
                "RETURN 1 AS test_value", database_=self.database
            )
            if records and records[0]["test_value"] == 1:
                logger.info(
                    f"Tier 2 verification passed: Database '{self.database}' is accessible"
                )
            else:
                raise ConnectionError(
                    "Database verification query returned unexpected result"
                )
        except ClientError as e:
            if "DatabaseNotFound" in str(e):
                logger.error(f"Database '{self.database}' not found")
            raise

    def connect(self) -> Driver:
        """Establish connection to Neo4j database."""
        return self.driver

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j driver connection closed")

    def is_connected(self) -> bool:
        """Check if the driver is connected and responsive."""
        if not self._driver:
            return False

        try:
            self._driver.verify_connectivity()
            return True
        except (ServiceUnavailable, Neo4jError):
            return False

    @contextmanager
    def session(self, **session_config):
        """Context manager for Neo4j sessions with automatic cleanup.

        Args:
            **session_config: Additional session configuration parameters.

        Yields:
            Neo4j Session object.
        """
        session = self.driver.session(database=self.database, **session_config)
        try:
            yield session
        finally:
            session.close()

    def execute_query(
        self, query: str, parameters: dict[str, Any] | None = None, **kwargs
    ) -> tuple[list[dict], Any, list[str]]:
        """Execute a Cypher query with retry logic.

        Args:
            query: Cypher query string.
            parameters: Query parameters.
            **kwargs: Additional query execution parameters.

        Returns:
            Tuple of (records, summary, keys).
        """

        @self.retry_on_transient_error
        def _execute():
            return self.driver.execute_query(
                query, parameters or {}, database_=self.database, **kwargs
            )

        return _execute()

    def execute_write_transaction(self, transaction_func, *args, **kwargs):
        """Execute a write transaction with retry logic.

        Args:
            transaction_func: Function to execute within transaction.
            *args, **kwargs: Arguments to pass to transaction function.

        Returns:
            Result from transaction function.
        """

        @self.retry_on_transient_error
        def _execute():
            with self.session() as session:
                return session.execute_write(transaction_func, *args, **kwargs)

        return _execute()

    def execute_read_transaction(self, transaction_func, *args, **kwargs):
        """Execute a read transaction with retry logic.

        Args:
            transaction_func: Function to execute within transaction.
            *args, **kwargs: Arguments to pass to transaction function.

        Returns:
            Result from transaction function.
        """

        @self.retry_on_transient_error
        def _execute():
            with self.session() as session:
                return session.execute_read(transaction_func, *args, **kwargs)

        return _execute()

    def test_connection(self) -> dict[str, Any]:
        """Test the database connection and return diagnostic information.

        Returns:
            Dictionary containing connection test results.
        """
        results = {
            "uri": self.uri,
            "database": self.database,
            "connected": False,
            "server_version": None,
            "database_exists": False,
            "error": None,
        }

        try:
            # Test basic connectivity
            self.driver.verify_connectivity()
            results["connected"] = True

            # Get server version
            records, _, _ = self.execute_query(
                "CALL dbms.components() YIELD name, versions RETURN name, versions"
            )
            for record in records:
                if record["name"] == "Neo4j Kernel":
                    results["server_version"] = (
                        record["versions"][0] if record["versions"] else "Unknown"
                    )

            # Test database accessibility
            records, _, _ = self.execute_query("RETURN 1 AS test")
            results["database_exists"] = len(records) > 0

            # Get database info
            records, _, _ = self.execute_query(
                "CALL db.info() YIELD name, creationDate RETURN name, creationDate"
            )
            if records:
                results["database_info"] = {
                    "name": records[0]["name"],
                    "creation_date": str(records[0]["creationDate"]),
                }

            logger.info(f"Connection test successful: {results}")

        except Exception as e:
            results["error"] = str(e)
            logger.error(f"Connection test failed: {e}")

        return results

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Singleton instance for convenience
_default_connection: Neo4jConnection | None = None


def get_connection() -> Neo4jConnection:
    """Get or create the default Neo4j connection instance.

    Returns:
        Neo4jConnection instance.
    """
    global _default_connection
    if _default_connection is None:
        _default_connection = Neo4jConnection()
    return _default_connection


def close_default_connection() -> None:
    """Close the default connection if it exists."""
    global _default_connection
    if _default_connection:
        _default_connection.close()
        _default_connection = None
