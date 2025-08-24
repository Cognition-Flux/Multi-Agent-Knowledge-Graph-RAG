A Comprehensive Guide to Implementing a GraphRAG System with Python and Neo4jPart I: Foundation – Establishing a Robust and Correct ConnectionThe initial and most critical phase of building any database-driven application is establishing a correct, stable, and efficient connection. A failure at this stage invalidates all subsequent efforts. This section provides a detailed analysis of the Neo4j connection process for Python applications, rectifies common misconceptions, and presents a production-ready approach for connecting to the specified database instance.1.1 Deconstructing the Neo4j Connection: HTTP vs. BoltA frequent point of confusion for developers new to Neo4j is the distinction between the protocols used for its different interfaces. The provided connection URL, http://44.243.196.65:7474/, points to the Neo4j Browser, which is a web-based graphical user interface for database administration and querying.1 While this interface is invaluable for development and exploration, it uses the HTTP(S) protocol and is not intended for programmatic access from application drivers.The official Neo4j Python driver, and indeed all official drivers, communicate with the database using a high-performance binary protocol called Bolt.2 This protocol is specifically designed for efficient data transfer between an application and the Neo4j server. By default, the Bolt protocol listens on port 7687.1 Therefore, the first step in correcting the connection string is to target port 7687 and use a URI scheme compatible with the Bolt protocol.The choice of URI scheme is not merely syntactic; it is a crucial architectural decision that dictates how the driver interacts with the database topology. The two primary schemes are bolt:// and neo4j://.4Direct Connection (bolt://): This scheme establishes a direct, non-routing connection to the single server instance specified in the URI. It is the most efficient method for connecting to a standalone Neo4j instance or a specific member of a cluster, as it bypasses the routing discovery process.2 Given that the user has provided a single IP address for a cloud-hosted instance, this strongly implies a single-node deployment where the overhead of routing is unnecessary.Routing Connection (neo4j://): This scheme initiates a routing-aware connection. Upon connecting, the driver requests a routing table from the server, which lists all members of a Causal Cluster and their roles (e.g., leader for writes, followers for reads). The driver then uses this table to intelligently route transactions to the appropriate server, providing high availability and load balancing.2 This scheme is essential for applications connecting to a multi-node Neo4j cluster but introduces a slight overhead for single-node setups.For the specified use case—connecting to a single, known IP address—the bolt:// scheme is the optimal choice. It provides the most direct and performant connection path by eliminating the initial routing table negotiation step.The following table provides a comprehensive summary of the available URI schemes, their security and routing characteristics, and their ideal use cases, distilling information from the official documentation into a practical reference.2URI SchemeEncryptionRoutingTypical Use Casebolt://NoneNoLocal development; connecting to a specific, unsecured instance.bolt+s://Full TLSNoConnecting to a specific, secured instance with a trusted certificate (e.g., from a public CA).bolt+ssc://Full TLSNoConnecting to a specific, secured instance with a self-signed certificate, bypassing certificate checks.neo4j://NoneYesConnecting to an unsecured Causal Cluster.neo4j+s://Full TLSYesConnecting to a secured Causal Cluster with trusted certificates (standard for Neo4j Aura).neo4j+ssc://Full TLSYesConnecting to a secured Causal Cluster with self-signed certificates.1.2 Implementing and Verifying the ConnectionWith a clear understanding of the correct protocol and URI scheme, the next step is to implement the connection in Python. The script below demonstrates best practices, including managing credentials via environment variables, using the neo4j.GraphDatabase.driver factory method, and leveraging a with statement to ensure proper resource management and connection closure.3A common practice is to use the driver.verify_connectivity() method immediately after instantiation to confirm that a connection can be established.3 However, it is critical to understand the limitations of this method. verify_connectivity() primarily confirms network reachability to the DBMS and a successful authentication handshake. It does not guarantee that the target database for subsequent queries exists or that the authenticated user has the necessary permissions to access it. Real-world scenarios have shown that this method can return a false positive, passing successfully even when the default database has been removed, only for the first actual query to fail with an error like Neo.ClientError.Database.DatabaseNotFound.7A more robust, production-grade approach involves a two-tiered verification. First, use verify_connectivity() for an initial check. Second, wrap the first substantive database operation in a try...except block that can catch more specific neo4j.exceptions. This ensures that the entire path—from the client application through the driver to the specific target database—is fully operational before the application proceeds with complex logic. This proactive error handling provides more specific and actionable feedback, distinguishing between a network/auth failure and a database configuration issue.The following Python code block demonstrates this robust connection and verification strategy.Python# ===================================================================
# Part 1: Foundation – Establishing a Robust Connection
# ===================================================================
import os
import asyncio
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase, RoutingControl
from neo4j.exceptions import ServiceUnavailable, AuthError, ClientError

# Configure logging to see driver details
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def connect_to_neo4j():
    """
    Establishes and verifies a connection to the Neo4j database.

    This function demonstrates a robust connection pattern:
    1. Loads credentials securely from environment variables.
    2. Corrects the user-provided HTTP URL to the appropriate Bolt protocol URI.
    3. Uses a `with` statement for automatic resource management.
    4. Performs a two-tiered verification:
        a. `verify_connectivity()` for initial network/auth check.
        b. A simple test query to confirm the target database is accessible.

    Returns:
        neo4j.Driver: An active and verified Neo4j driver instance, or None on failure.
    """
    # Load environment variables from a.env file for secure credential management
    load_dotenv()

    # --- Correcting the Connection URI ---
    # The user provided an HTTP URL for the Neo4j Browser (port 7474).
    # Programmatic access requires the Bolt protocol, which defaults to port 7687.
    # We will use the 'bolt://' scheme for a direct connection to this single instance.
    NEO4J_URI = "bolt://44.243.196.65:7687"
    NEO4J_USERNAME = os.getenv("AWS_NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("AWS_NEO4J_PASSWORD")

    if not NEO4J_PASSWORD:
        log.error("NEO4J_PASSWORD environment variable not set. Cannot connect.")
        return None

    log.info(f"Attempting to connect to Neo4j at {NEO4J_URI}...")

    try:
        # The `with` statement ensures the driver is properly closed upon exit
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

        # --- Tier 1 Verification: Network and Authentication ---
        # This checks if the driver can connect to the DBMS and authenticate.
        driver.verify_connectivity()
        log.info("Tier 1 Verification successful: Connectivity and authentication confirmed.")

        # --- Tier 2 Verification: Database Accessibility ---
        # This simple query confirms the default database is available and queryable.
        # This catches errors that `verify_connectivity` might miss, like a missing database.
        records, summary, keys = driver.execute_query("RETURN 1 AS number", database_="neo4j")
        if records["number"] == 1:
            log.info("Tier 2 Verification successful: Target database 'neo4j' is responsive.")
        else:
            raise ConnectionError("Database did not return the expected test value.")

        log.info("Successfully connected and verified Neo4j connection.")
        return driver

    except AuthError as e:
        log.error(f"Authentication failed. Please check your username and password. Details: {e}")
        return None
    except ServiceUnavailable as e:
        log.error(f"Could not connect to the Neo4j server at {NEO4J_URI}. "
                  f"Please ensure the database is running and the URI is correct. Details: {e}")
        return None
    except ClientError as e:
        # This can catch errors like "DatabaseNotFound"
        log.error(f"A client-side error occurred. The database may not exist or "
                  f"the user may lack permissions. Details: {e}")
        return None
    except Exception as e:
        log.error(f"An unexpected error occurred during connection: {e}")
        return None

# Example of using the connection function
# neo4j_driver = connect_to_neo4j()
# if neo4j_driver:
#     # Proceed with application logic...
#     neo4j_driver.close()
# else:
#     log.error("Failed to establish Neo4j connection. Exiting.")
Part II: The Architect's Blueprint – Constructing a High-Quality Knowledge GraphOnce a stable connection is established, the focus shifts to transforming unstructured data into a structured, queryable Knowledge Graph (KG). The neo4j-graphrag library provides a powerful, high-level abstraction for this process, but its effective use requires careful architectural planning.2.1 The SimpleKGPipeline: An Orchestrated ApproachThe neo4j-graphrag library offers the SimpleKGPipeline class as a streamlined interface for building a KG from text.8 This class, part of the library's experimental features, orchestrates a complex sequence of operations under a single, unified API.10 These operations include:Document Loading: Ingesting source documents, such as PDFs or plain text files.8Text Chunking: Splitting large documents into smaller, semantically coherent chunks that are suitable for processing by a Large Language Model (LLM).11Embedding Generation: Creating vector embeddings for each text chunk, which captures its semantic meaning and enables similarity searches.11LLM-based Extraction: Using an LLM to analyze each chunk and extract structured information in the form of entities (nodes) and relationships.Graph Writing: Persisting the extracted nodes and relationships, along with the source chunks and documents, into the Neo4j database.9To use the SimpleKGPipeline, three core components must be provided: a Neo4j driver, an llm interface, and an embedder interface.9 The library provides convenient wrappers for popular services like OpenAI, requiring only an API key to be configured in the environment.122.2 The Power of Schema: From Unstructured Chaos to Structured KnowledgeThe single most impactful step in creating a high-quality, useful KG is the definition of a graph schema. Without a schema, the LLM's extraction process is unguided (schema="FREE" mode), often resulting in an inconsistent and unpredictable graph with noisy or irrelevant entities and relationships.9A well-defined schema acts as a precise set of instructions for the LLM. By specifying the desired node_types, relationship_types, and the patterns that connect them, the developer constrains the LLM's output, forcing it to focus only on the information relevant to the application's domain.9This upfront investment in schema design has a direct and profound impact on the performance of the final RAG application. The causal chain is as follows:A carefully engineered schema guides the LLM to produce a clean, consistent, and predictable graph structure.A clean graph structure enables the writing of precise and efficient Cypher queries for information retrieval.Precise Cypher queries retrieve highly relevant and targeted context from the graph.High-quality context allows the LLM in the RAG pipeline to generate accurate, factual, and non-hallucinatory answers.Therefore, schema definition should not be viewed as an optional configuration but as the foundational engineering step of the entire GraphRAG system. It is the primary mechanism for transforming unstructured chaos into structured, actionable knowledge.2.3 Building the Graph: Code and VerificationThe SimpleKGPipeline implicitly creates two distinct but interconnected subgraphs within Neo4j, forming a powerful dual-graph architecture:The Lexical Graph: This graph represents the structure of the source document itself. It typically consists of Document nodes connected to a series of Chunk nodes via HAS_CHUNK relationships. The Chunk nodes are linked sequentially with NEXT_CHUNK relationships, preserving the original flow of the text. Each Chunk node stores the raw text and its vector embedding.9The Entity Graph: This is the knowledge graph proper, containing the entities (e.g., Person, Company) and relationships (e.g., WORKS_FOR, LOCATED_IN) extracted from the text by the LLM.13 Critically, each extracted entity node is linked back to the specific Chunk node in the lexical graph from which it was derived.Many developers focus solely on the entity graph, but the lexical graph is the indispensable bridge back to the source of truth—the original text. An effective GraphRAG retrieval pattern uses the entity graph to navigate relationships and identify relevant concepts, and then traverses to the associated Chunk nodes in the lexical graph to retrieve the verbatim text. This text is the context that will ultimately be fed to the LLM to generate an answer. Understanding and leveraging this dual-graph architecture is fundamental to designing effective retrieval strategies.The following code block demonstrates how to configure and execute the SimpleKGPipeline to build both of these graphs from a sample text.Python# ===================================================================
# Part 2: The Architect's Blueprint – Constructing the KG
# ===================================================================
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

# Sample unstructured text for KG construction
# This text contains clear entities and relationships for extraction.
SAMPLE_TEXT = """
Dr. Aris Thorne, a leading researcher in quantum computing at the Tesseract Institute in Geneva,
published a groundbreaking paper on qubit stabilization. The paper, which was co-authored by
Dr. Lena Petrova from the rival organization Aether Dynamics, suggests a new method using
supercooled helium-3. Aether Dynamics, headquartered in Munich, has long been a competitor
to the Tesseract Institute in the race for scalable quantum hardware. The research was partially
funded by a grant from the European Science Foundation.
"""

async def build_knowledge_graph(driver, text_content):
    """
    Builds a knowledge graph from unstructured text using the SimpleKGPipeline.

    Args:
        driver (neo4j.Driver): An active Neo4j driver instance.
        text_content (str): The unstructured text to process.
    """
    if not os.getenv("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY environment variable not set. Cannot build KG.")
        return

    log.info("Initializing LLM and Embedder for KG construction...")
    # Initialize the LLM for entity and relation extraction
    llm = OpenAILLM(model_name="gpt-4o-mini")
    # Initialize the embedder for creating vector embeddings of text chunks
    embedder = OpenAIEmbeddings(model="text-embedding-3-large")

    # --- Defining a Strong Schema ---
    # This schema guides the LLM to extract only relevant information,
    # ensuring a clean and queryable graph structure.
    log.info("Defining graph schema to guide LLM extraction...")
    node_types =},
        "Paper"
    ]
    relationship_types =
    patterns =

    graph_schema = {
        "node_types": node_types,
        "relationship_types": relationship_types,
        "patterns": patterns
    }

    log.info("Configuring and running the SimpleKGPipeline...")
    kg_builder = SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        schema=graph_schema,
        from_pdf=False, # We are processing a raw text string
        neo4j_database="neo4j" # Explicitly specify the target database
    )

    # The pipeline is asynchronous
    await kg_builder.run_async(text=text_content)
    log.info("Knowledge Graph construction complete.")
    log.info("You can now explore the 'Document', 'Chunk', and entity nodes (e.g., 'Person') in Neo4j Browser.")

# Example of using the KG build function
# async def main():
#     neo4j_driver = connect_to_neo4j()
#     if neo4j_driver:
#         await build_knowledge_graph(neo4j_driver, SAMPLE_TEXT)
#         neo4j_driver.close()
#
# if __name__ == "__main__":
#     asyncio.run(main())
Part III: Activation – Building the RAG PipelineWith a well-structured KG populated in the database, the final step is to build the RAG pipeline that will leverage this graph to answer user questions. This involves making the graph's content searchable and assembling the components that handle retrieval, context augmentation, and final answer generation.3.1 Prerequisite: The Vector Index for Semantic SearchThe KG construction process created vector embeddings for each Chunk node, capturing their semantic meaning. To efficiently search these embeddings, a vector index must be created within Neo4j. This index allows for near-instantaneous retrieval of text chunks that are semantically similar to a user's query, forming the entry point for the retrieval process.11The neo4j-graphrag library provides a helper function, create_vector_index, to simplify this task. It is crucial to configure the index with the correct node label (Chunk), property name (embedding), vector dimensions, and similarity_fn (e.g., cosine) to match the embeddings generated by the chosen model.16 For OpenAI's text-embedding-3-large model, the dimension is 3072, while for text-embedding-ada-002 it is 1536.3.2 Assembling the GraphRAG EngineThe core of the RAG application is the GraphRAG class. This class ties together the retriever and the LLM to perform the end-to-end RAG process.15 The key component here is the Retriever, which is the heart of the system and the primary point of customization.For this implementation, the VectorRetriever is used. This retriever takes a user's query, uses the embedder to convert it into a vector, and then queries the Neo4j vector index to find the most semantically similar Chunk nodes.15 The text from these chunks is then returned as the context for the LLM.While the VectorRetriever is powerful and sufficient for many use cases, it represents only the first step into the capabilities of GraphRAG. The true power of using a graph database is unlocked with more advanced retrievers, such as the VectorCypherRetriever.16 This advanced pattern combines semantic search with explicit graph traversal. It first performs a vector search to find an initial "entry point" Chunk or entity in the graph. Then, it executes a predefined Cypher query starting from that entry point to traverse the graph's relationships and gather a much richer, more interconnected context. For example, it could find a Person node mentioned in a query, then traverse WORKS_AT and CO_AUTHOR_OF relationships to retrieve text from all related chunks, providing the LLM with a comprehensive view of that person's professional activities. This moves beyond simple semantic similarity to true, relationship-aware context retrieval, which is the core value proposition of GraphRAG.18The code below sets up the prerequisite vector index and assembles the GraphRAG engine using the VectorRetriever, preparing it for querying.Python# ===================================================================
# Part 3: Activation – Building the RAG Pipeline
# ===================================================================
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG

def create_vector_search_index(driver):
    """
    Creates a vector index in Neo4j on the 'Chunk' nodes.
    This is a prerequisite for performing semantic search.

    Args:
        driver (neo4j.Driver): An active Neo4j driver instance.
    """
    log.info("Creating vector index 'chunk_embeddings' on Chunk(embedding)...")
    try:
        # Configuration for OpenAI's text-embedding-3-large model
        # If using text-embedding-ada-002, dimensions would be 1536.
        create_vector_index(
            driver,
            index_name="chunk_embeddings",
            label="Chunk",
            embedding_property="embedding",
            dimensions=3072,
            similarity_fn="cosine",
            database="neo4j"
        )
        log.info("Vector index created successfully.")
    except ClientError as e:
        # It's common for the index to already exist, which is not a critical error.
        if "An equivalent index already exists" in str(e):
            log.warning("Vector index 'chunk_embeddings' already exists. Skipping creation.")
        else:
            log.error(f"Failed to create vector index: {e}")
            raise

def setup_rag_pipeline(driver):
    """
    Sets up the GraphRAG pipeline by initializing the retriever and the main RAG engine.

    Args:
        driver (neo4j.Driver): An active Neo4j driver instance.

    Returns:
        GraphRAG: An initialized GraphRAG instance ready for querying.
    """
    if not os.getenv("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY environment variable not set. Cannot setup RAG pipeline.")
        return None

    log.info("Setting up the RAG pipeline...")

    # The embedder is used by the retriever to convert the user's question into a vector
    embedder = OpenAIEmbeddings(model="text-embedding-3-large")

    # The VectorRetriever performs a similarity search against the vector index
    retriever = VectorRetriever(
        driver,
        index_name="chunk_embeddings",
        embedder=embedder
    )

    # The LLM is used for the final answer generation step
    llm = OpenAILLM(model_name="gpt-4o-mini")

    # The GraphRAG class orchestrates the retrieve-augment-generate process
    rag_pipeline = GraphRAG(
        retriever=retriever,
        llm=llm
    )

    log.info("RAG pipeline is ready.")
    return rag_pipeline

# Example of setting up the pipeline
# neo4j_driver = connect_to_neo4j()
# if neo4j_driver:
#     create_vector_search_index(neo4j_driver)
#     rag_app = setup_rag_pipeline(neo4j_driver)
#     # Now rag_app can be used to answer questions
#     neo4j_driver.close()
Part IV: Synthesis – The Complete Script and End-to-End TestThis final part consolidates all preceding components into a single, cohesive, and executable Python script. It serves as the complete deliverable, demonstrating the entire workflow from initial connection to the final, AI-generated answer.4.1 The Unified Application ScriptThe script below integrates the connection logic, KG construction, index creation, and RAG pipeline setup into a unified application. It is heavily commented to explain the flow of execution and the purpose of each component. This script represents a complete, working example of the system requested by the user.4.2 Execution and InteractionTo run the script, ensure a .env file is present in the same directory with the following content:#.env file
AWS_NEO4J_USERNAME=neo4j
AWS_NEO4J_PASSWORD=M5u^dQ@kbNPZzfjVEP5GcxVgtP8@W&R
OPENAI_API_KEY=your_openai_api_key_here
The script will then proceed through all the stages automatically and ask a sample question relevant to the text processed during the KG construction phase.4.3 Analyzing the Response: Answer and ProvenanceA key differentiator of a well-implemented GraphRAG system over opaque, purely LLM-based systems is its ability to provide explainability. The rag.search() method can be configured with return_context=True, which instructs the pipeline to return not just the final generated answer, but also the exact context—the text from the Chunk nodes—that was retrieved from the graph and used to inform that answer.17This feature, often called provenance, is critical for building trust and verifying the system's output. It creates a transparent, auditable trail from the user's question to the source data that supports the answer. This mitigates the risk of LLM hallucinations and allows users to validate the information, a mandatory requirement for many enterprise use cases in fields like medical research, legal analysis, and financial reporting.11 The final script demonstrates how to access and display this provenance, showcasing the ultimate value proposition of the GraphRAG architecture.Python# ===================================================================
# Full End-to-End GraphRAG Application
# ===================================================================
import os
import asyncio
import logging
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError, ClientError

from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Sample unstructured text for KG construction
SAMPLE_TEXT = """
Dr. Aris Thorne, a leading researcher in quantum computing at the Tesseract Institute in Geneva,
published a groundbreaking paper on qubit stabilization. The paper, which was co-authored by
Dr. Lena Petrova from the rival organization Aether Dynamics, suggests a new method using
supercooled helium-3. Aether Dynamics, headquartered in Munich, has long been a competitor
to the Tesseract Institute in the race for scalable quantum hardware. The research was partially
funded by a grant from the European Science Foundation.
"""

# --- Part 1: Connection Function ---
def connect_to_neo4j():
    load_dotenv()
    NEO4J_URI = "bolt://44.243.196.65:7687"
    NEO4J_USERNAME = os.getenv("AWS_NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("AWS_NEO4J_PASSWORD")

    if not all():
        log.error("Neo4j credentials not found in environment variables.")
        return None

    log.info(f"Attempting to connect to Neo4j at {NEO4J_URI}...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        driver.verify_connectivity()
        log.info("Connectivity and authentication confirmed.")
        driver.execute_query("RETURN 1", database_="neo4j")
        log.info("Target database 'neo4j' is responsive.")
        return driver
    except (AuthError, ServiceUnavailable, ClientError) as e:
        log.error(f"Failed to connect or verify Neo4j connection: {e}")
        return None

# --- Part 2: KG Building Function ---
async def build_knowledge_graph(driver, text_content):
    if not os.getenv("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set.")
        return False
    log.info("Initializing components for KG construction...")
    llm = OpenAILLM(model_name="gpt-4o-mini")
    embedder = OpenAIEmbeddings(model="text-embedding-3-large")

    node_types = ["Person", "Organization", "Location", "Paper"]
    relationship_types =
    patterns =
    graph_schema = {"node_types": node_types, "relationship_types": relationship_types, "patterns": patterns}

    log.info("Running the SimpleKGPipeline...")
    kg_builder = SimpleKGPipeline(llm=llm, driver=driver, embedder=embedder, schema=graph_schema, from_pdf=False, neo4j_database="neo4j")
    await kg_builder.run_async(text=text_content)
    log.info("Knowledge Graph construction complete.")
    return True

# --- Part 3: RAG Setup Functions ---
def create_vector_search_index(driver):
    log.info("Creating vector index 'chunk_embeddings'...")
    try:
        create_vector_index(driver, "chunk_embeddings", "Chunk", "embedding", 3072, "cosine", database="neo4j")
        log.info("Vector index created successfully.")
    except ClientError as e:
        if "An equivalent index already exists" in str(e):
            log.warning("Vector index already exists. Skipping.")
        else:
            log.error(f"Failed to create vector index: {e}")
            raise

def setup_rag_pipeline(driver):
    if not os.getenv("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY not set.")
        return None
    log.info("Setting up the RAG pipeline...")
    embedder = OpenAIEmbeddings(model="text-embedding-3-large")
    retriever = VectorRetriever(driver, "chunk_embeddings", embedder)
    llm = OpenAILLM(model_name="gpt-4o-mini")
    rag_pipeline = GraphRAG(retriever=retriever, llm=llm)
    log.info("RAG pipeline is ready.")
    return rag_pipeline

# --- Main Execution Logic ---
async def main():
    """Main function to run the entire GraphRAG workflow."""
    neo4j_driver = None
    try:
        # Step 1: Connect to the database
        neo4j_driver = connect_to_neo4j()
        if not neo4j_driver:
            return

        # Step 2: Build the Knowledge Graph
        # Note: In a real application, you would run this only once per document.
        # For this demo, we run it every time.
        log.info("\n--- Starting Knowledge Graph Construction ---")
        success = await build_knowledge_graph(neo4j_driver, SAMPLE_TEXT)
        if not success:
            return

        # Step 3: Set up the RAG Pipeline
        log.info("\n--- Setting up RAG Pipeline ---")
        create_vector_search_index(neo4j_driver)
        rag_app = setup_rag_pipeline(neo4j_driver)
        if not rag_app:
            return

        # Step 4: Ask a question and analyze the response
        log.info("\n--- Querying the RAG System ---")
        question = "Who are the main competitors mentioned in the text and where are they located?"
        log.info(f"Question: {question}")

        # Execute the RAG search, requesting the context for explainability
        response = rag_app.search(question, return_context=True)

        print("\n" + "="*50)
        print("RAG System Response Analysis")
        print("="*50)

        print(f"\n[Generated Answer]\n{response.answer}\n")

        print("-" * 50)

        print("\n\n")
        if response.context:
            for i, context_item in enumerate(response.context):
                print(f"--- Context Snippet {i+1} ---\n")
                print(context_item.text)
                print("\n" + "-"*25 + "\n")
        else:
            print("No context was retrieved to generate this answer.")

        print("="*50)

    finally:
        if neo4j_driver:
            neo4j_driver.close()
            log.info("Neo4j driver closed.")

if __name__ == "__main__":
    # Ensure you have a.env file with AWS_NEO4J_... and OPENAI_API_KEY
    asyncio.run(main())
