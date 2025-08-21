"""Knowledge Graph Search module for Hybrid GraphRAG Agent.

This module configures the GraphRAG instance with the appropriate
LLM, retriever, and prompt template for searching the knowledge graph.
"""

# %%
import logging
import os

from dotenv import load_dotenv
from langchain_aws import ChatBedrock, ChatBedrockConverse
from neo4j_graphrag.generation import GraphRAG, RagTemplate
from neo4j_graphrag.llm import AzureOpenAILLM

from src.agents.hybrid_graphRAG_agent.bedrock_converse_adapter import (
    BedrockConverseLLMAdapter,
)
from src.agents.hybrid_graphRAG_agent.retriever import retriever


# --- Setup ---
load_dotenv(override=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configure LLM
LLM_PROVIDER: str = "bedrock"  # bedrock
BEDROCK_CHAT_MODEL: str = "CONVERSE"

# Ensure we always define llm; raise if unsupported provider
LLM = None

if LLM_PROVIDER == "azure":
    try:
        LLM = AzureOpenAILLM(
            model_name="gpt-4.1",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
    except Exception as exc:
        logging.error("Failed to initialize Azure OpenAI LLM: %s", exc)
        raise

elif LLM_PROVIDER == "bedrock":
    if BEDROCK_CHAT_MODEL == "CONVERSE":
        try:
            CONVERSE_MODEL = "us.anthropic.claude-sonnet-4-20250514-v1:0"
            bedrock_chat = ChatBedrockConverse(
                model=CONVERSE_MODEL,
                region_name=os.getenv("AWS_BEDROCK_REGION"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.getenv("AWS_BEARER_TOKEN_BEDROCK"),
            )
            LLM = BedrockConverseLLMAdapter(
                chat=bedrock_chat, model_name=CONVERSE_MODEL
            )
        except Exception as exc:
            logging.error("Failed to initialize Bedrock LLM: %s", exc)
            raise
    else:
        try:
            LLM = ChatBedrock(
                model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
                region_name=os.getenv("AWS_BEDROCK_REGION"),
            )
        except Exception as exc:
            logging.error("Failed to initialize Bedrock LLM: %s", exc)
            raise
else:
    raise ValueError(
        f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Use 'azure' or 'bedrock'."
    )
print(f"LLM Provider: {LLM_PROVIDER}")
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
    graph_rag = GraphRAG(retriever=retriever, llm=LLM, prompt_template=rag_template)
    logging.info("GraphRAG instance successfully initialized")
except Exception as exc:
    logging.error("Failed to initialize GraphRAG: %s", exc)
    raise

if __name__ == "__main__":
    # Test the GraphRAG instance with a sample query
    QUERY = (
        "¿Qué información tienes sobre el proyecto de energía solar "
        "en la región de Coquimbo?"
    )

    print("\n🔍 Testing GraphRAG Search")
    print(f"📝 Query: {QUERY}\n")

    response = graph_rag.search(
        QUERY,
        retriever_config={"top_k": 10},
        return_context=False,
    )
    print("✅ Answer:")
    print(response.answer)
