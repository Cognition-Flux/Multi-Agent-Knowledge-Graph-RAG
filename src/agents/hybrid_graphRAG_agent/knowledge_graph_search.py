"""Knowledge Graph Search module for Hybrid GraphRAG Agent.

This module configures the GraphRAG instance with the appropriate
LLM, retriever, and prompt template for searching the knowledge graph.
"""

# %%
import logging
import os

from dotenv import load_dotenv
from neo4j_graphrag.generation import GraphRAG, RagTemplate
from neo4j_graphrag.llm import AzureOpenAILLM

from src.agents.hybrid_graphRAG_agent.retriever import retriever


# --- Setup ---
load_dotenv(override=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configure LLM
try:
    llm = AzureOpenAILLM(
        model_name="gpt-4.1",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
except Exception as exc:
    logging.error("Failed to initialize Azure OpenAI LLM: %s", exc)
    raise

# RAG prompt template
rag_template = RagTemplate(
    template="""You are an expert in environmental impact assessment projects.
Answer the **Question** ONLY using the **Context** provided.

IMPORTANT RULES:
- NEVER add NOR inject information or data that is not in the context.
- If the context doesn't contain relevant information, clearly state that.
- Be concise and precise in your answers.

# Question:
{query_text}

# Context:
{context}

# Answer:
""",
    expected_inputs=["query_text", "context"],
)

# Create GraphRAG instance with the configured components
try:
    graph_rag = GraphRAG(retriever=retriever, llm=llm, prompt_template=rag_template)
    logging.info("GraphRAG instance successfully initialized")
except Exception as exc:
    logging.error("Failed to initialize GraphRAG: %s", exc)
    raise

if __name__ == "__main__":
    # Test the GraphRAG instance with a sample query
    QUERY = "¿Qué información tienes sobre el CENTRO DE RECEPCIÓN Y DISPOSICIÓN FINAL DE BIOSÓLIDOS?"

    print("\n🔍 Testing GraphRAG Search")
    print(f"📝 Query: {QUERY}\n")

    try:
        response = graph_rag.search(
            QUERY,
            retriever_config={"top_k": 10},
            return_context=False,
        )
        print("✅ Answer:")
        print(response.answer)
    except Exception as exc:
        print(f"❌ Error: {exc}")
