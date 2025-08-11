"""graphRAG agent logic.

This file contains the logic for the hybrid graphRAG agent.
"""

# %%
from __future__ import annotations

import asyncio
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send
from neo4j_graphrag.generation import GraphRAG, RagTemplate
from neo4j_graphrag.llm import AzureOpenAILLM
from pydantic import BaseModel, Field

from src.agents.cypher_query_agent.llm_chains import get_question_generation_chain
from src.agents.cypher_query_agent.reducers import reduce_lists
from src.agents.cypher_query_agent.schemas import GeneratedQueries
from src.agents.hybrid_graphRAG_agent.hybrid_cypher_retriever import retriever


# --------------------------------------------------------------------------- #
# 1) Entorno e índices
# --------------------------------------------------------------------------- #

load_dotenv(override=True)

llm = AzureOpenAILLM(
    model_name="gpt-4.1",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

rag_template = RagTemplate(
    template="""You are a metabolic pathway expert. Answer the **Question** ONLY
 using the **Context** provided.

 NEVER add NOR inject information or data that is not in the context.

 # Question:
 {query_text}

 # Context:
 {context}

 # Answer:
 """,
    expected_inputs=["query_text", "context"],
)

# --------------------------------------------------------------------------- #
# 2) GraphRAG pipeline
# --------------------------------------------------------------------------- #

graph_rag = GraphRAG(retriever=retriever, llm=llm, prompt_template=rag_template)


# --------------------------------------------------------------------------- #
# 3) State Schema
# --------------------------------------------------------------------------- #


class GraphRAGQueryState(BaseModel):
    """State of the Hybrid GraphRAG Agent."""

    question: str = Field(default="")
    generated_questions: GeneratedQueries | None = Field(default=None)
    query: str = Field(default="")
    results: Annotated[list[str], reduce_lists] = Field(default_factory=list)
    messages: list[AIMessage] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# 4) LLM Chains
# --------------------------------------------------------------------------- #

# Reutilizamos la cadena de generación de preguntas del cypher_query_agent
qgen_chain = get_question_generation_chain(group="FEW_SHOTS_QUESTIONS_GENERATION", k=2)


# --------------------------------------------------------------------------- #
# 5) Node Functions
# --------------------------------------------------------------------------- #


async def generate_questions(
    state: GraphRAGQueryState,
) -> Command[Literal["send_queries_in_parallel"]]:
    """Node that generates multiple related queries from the original question."""
    print(f"\n📝 Generating related questions for: {state.question}")

    generated_questions = await qgen_chain.ainvoke({"input": state.question})

    print(f"✅ Generated {len(generated_questions.queries_list)} questions")
    for i, q in enumerate(generated_questions.queries_list, 1):
        print(f"   {i}. {q.query_str}")
    return Command(
        goto="send_queries_in_parallel",
        update={"generated_questions": generated_questions},
    )


async def send_queries_in_parallel(
    state: GraphRAGQueryState,
) -> Command[list[Send]]:
    """Node that sends generated queries in parallel to GraphRAG."""
    if not state.generated_questions or not state.generated_questions.queries_list:
        # If no questions were generated, send the original question
        print("\n⚠️ No questions generated, using original question")
        sends = [Send("generate_answer", {"query": state.question})]
    else:
        lista_de_queries = [
            query.query_str for query in state.generated_questions.queries_list
        ]
        print(f"\n🚀 Sending {len(lista_de_queries)} queries in parallel to GraphRAG")
        sends = [
            Send(
                "generate_answer",
                {"query": query},
            )
            for query in lista_de_queries
        ]
    return Command(goto=sends)


async def generate_answer(
    state: GraphRAGQueryState | dict,
) -> Command[Literal["aggregate_results"]]:
    """Node that generates an answer using GraphRAG for a single query."""
    # Handle both dict (from Send) and GraphRAGQueryState
    if isinstance(state, dict):
        query_str = state.get("query", "")
    else:
        query_str = state.query

    print(f"\n🔍 Processing query: {query_str}")

    try:
        # Usar GraphRAG para buscar la respuesta
        response = graph_rag.search(
            query_str,
            retriever_config={"top_k": 5},
            return_context=False,
        )

        answer = response.answer
        print(f"✅ Answer obtained: {answer[:100]}...")

    except Exception as e:
        answer = f"Error processing query '{query_str}': {e!s}"
        print(f"❌ Error: {answer}")

    return Command(goto="aggregate_results", update={"results": [answer]})


async def aggregate_results(
    state: GraphRAGQueryState,
) -> Command[Literal[END]]:
    """Node that aggregates all results and creates the final response."""
    results = state.results
    question = state.question

    print(f"\n📊 Aggregating {len(results)} results")

    # Crear una respuesta consolidada
    if results:
        # Formato estructurado de las respuestas
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(f"**Perspectiva {i}:**\n{result}")

        final_answer = (
            f"Para responder a tu pregunta: '{question}', "
            f"he analizado {len(results)} perspectivas diferentes:\n\n"
            + "\n\n".join(formatted_results)
        )
    else:
        final_answer = f"No pude obtener información para responder: '{question}'"

    print(f"\n✅ Final answer prepared ({len(final_answer)} chars)")

    return Command(goto=END, update={"messages": [AIMessage(content=final_answer)]})


# --------------------------------------------------------------------------- #
# 6) Build Graph
# --------------------------------------------------------------------------- #


def build_graph() -> StateGraph:
    """Build and compile the GraphRAG workflow graph."""
    builder = StateGraph(GraphRAGQueryState)

    # Add nodes
    builder.add_node("generate_questions", generate_questions)
    builder.add_node("send_queries_in_parallel", send_queries_in_parallel)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("aggregate_results", aggregate_results)

    # Add edges
    builder.add_edge(START, "generate_questions")

    # Compile and return
    return builder.compile()


# Create the compiled graph
graph = build_graph()


# --------------------------------------------------------------------------- #
# 7) Main Execution (for testing)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    async def main():
        """Test the GraphRAG workflow with a sample question."""
        test_question = "¿cuáles son las comunas en los proyectos?"

        print("=" * 60)
        print("🤖 GraphRAG Agent Test")
        print(f"Question: {test_question}")
        print("=" * 60)

        try:
            # Run the graph
            async for _chunk in graph.astream(
                {"question": test_question},
                stream_mode="updates",
                subgraphs=True,
                debug=True,
            ):
                # The debug flag will print detailed execution info
                pass

            print("\n" + "=" * 60)
            print("✅ Workflow completed successfully!")
            print("=" * 60)

        except Exception as e:
            print(f"\n❌ Error running workflow: {e}")
            import traceback

            traceback.print_exc()

    # Run the async main function
    asyncio.run(main())
