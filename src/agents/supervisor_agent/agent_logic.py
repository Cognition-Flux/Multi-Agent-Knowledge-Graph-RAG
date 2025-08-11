"""Supervisor agent logic for routing questions to appropriate subgraphs.

This module implements a supervisor agent that decides whether to route
questions to the Cypher query agent (for metadata queries) or the hybrid
GraphRAG agent (for content queries).
"""

# %%
from __future__ import annotations

import asyncio
from enum import Enum
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

# Import the compiled subgraphs
from src.agents.cypher_query_agent.graph_builder import graph as cypher_graph
from src.agents.hybrid_graphRAG_agent.agent_logic import graph as graphrag_graph
from src.utils import get_llm


# --------------------------------------------------------------------------- #
# 1) Environment Setup
# --------------------------------------------------------------------------- #

load_dotenv(override=True)


# --------------------------------------------------------------------------- #
# 2) State Schema
# --------------------------------------------------------------------------- #


class SupervisorState(MessagesState):
    """State for the supervisor agent.

    Extends MessagesState to maintain message history and includes
    the question field for routing decisions.
    """

    question: str = Field(default="")
    final_answer: str = Field(default="")


# --------------------------------------------------------------------------- #
# 3) Routing Decision Schema
# --------------------------------------------------------------------------- #


class AgentType(str, Enum):
    """Enum for agent routing decisions."""

    CYPHER = "cypher_agent"
    GRAPHRAG = "graphrag_agent"


class RoutingDecision(BaseModel):
    """Schema for the supervisor's routing decision."""

    agent: AgentType = Field(description="Which agent to route the question to")
    reasoning: str = Field(description="Brief explanation of why this agent was chosen")


# --------------------------------------------------------------------------- #
# 4) Supervisor Decision Chain
# --------------------------------------------------------------------------- #

SUPERVISOR_SYSTEM_PROMPT = """You are a routing supervisor that decides which specialized agent
should handle a user's question about environmental projects.

You have two specialized agents available:

1. **cypher_agent**: Handles queries about PROJECT METADATA
   - Use for: counting projects, listing project names, finding projects by location (communes, regions)
   - Examples: "How many projects are there?", "What projects are in Antofagasta?",
     "List all communes with projects", "Which regions have mining projects?"

2. **graphrag_agent**: Handles queries about PROJECT CONTENT
   - Use for: species information, flora/fauna details, environmental impacts, project summaries
   - Examples: "What species are found in the projects?", "Summarize the environmental impact",
     "What flora species are endangered?", "Describe the vegetation in the area"

Analyze the user's question and route it to the appropriate agent.

IMPORTANT:
- Questions about numbers, names, or locations of projects → cypher_agent
- Questions about content, species, or details within projects → graphrag_agent
"""


def get_supervisor_chain():
    """Create the supervisor decision chain with structured output."""
    llm = get_llm().bind(temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SUPERVISOR_SYSTEM_PROMPT),
            (
                "human",
                "Question: {question}\n\nWhich agent should handle this question?",
            ),
        ]
    )

    chain = prompt | llm.with_structured_output(RoutingDecision)
    return chain.with_retry(stop_after_attempt=3)


# Cache the supervisor chain
supervisor_chain = get_supervisor_chain()


# --------------------------------------------------------------------------- #
# 5) Node Functions
# --------------------------------------------------------------------------- #


async def supervisor_decision(
    state: SupervisorState,
) -> Command[Literal["cypher_agent_wrapper", "graphrag_agent_wrapper"]]:
    """Supervisor node that decides which agent to route to."""
    question = state.get("question", "")

    if not question:
        # Extract question from messages if not in state
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, HumanMessage):
                question = msg.content
                break

    if not question:
        raise ValueError("No question found in state or messages")

    print(f"\n🎯 Supervisor analyzing question: {question}")

    # Get routing decision from LLM
    decision = await supervisor_chain.ainvoke({"question": question})

    print(f"📋 Decision: Route to {decision.agent.value}")
    print(f"💭 Reasoning: {decision.reasoning}")

    # Update state with the question and route to appropriate agent
    if decision.agent == AgentType.CYPHER:
        goto = "cypher_agent_wrapper"
    else:
        goto = "graphrag_agent_wrapper"

    return Command(goto=goto, update={"question": question})


async def cypher_agent_wrapper(state: SupervisorState) -> Command[Literal[END]]:
    """Wrapper for the Cypher query agent subgraph.

    Transforms SupervisorState to the format expected by the Cypher agent,
    invokes the subgraph, and transforms the response back.
    """
    print("\n🔄 Routing to Cypher Query Agent...")

    question = state["question"]

    try:
        # Invoke the Cypher subgraph with the question
        # The cypher graph expects a Neo4jQueryState with 'question' field
        result = await cypher_graph.ainvoke(
            {"question": question}, config={"recursion_limit": 50}
        )

        # Extract the answer from the result
        # The cypher agent returns messages in its state
        if result.get("messages"):
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                answer = last_message.content
            else:
                answer = str(last_message)
        else:
            answer = "No answer generated by Cypher agent"

        print(f"✅ Cypher Agent completed. Answer length: {len(answer)} chars")

    except Exception as e:
        answer = f"Error in Cypher agent: {e!s}"
        print(f"❌ Cypher Agent error: {e}")

    # Return the answer as an AI message
    return Command(
        goto=END,
        update={"messages": [AIMessage(content=answer)], "final_answer": answer},
    )


async def graphrag_agent_wrapper(state: SupervisorState) -> Command[Literal[END]]:
    """Wrapper for the hybrid GraphRAG agent subgraph.

    Transforms SupervisorState to the format expected by the GraphRAG agent,
    invokes the subgraph, and transforms the response back.
    """
    print("\n🔄 Routing to Hybrid GraphRAG Agent...")

    question = state["question"]

    try:
        # Invoke the GraphRAG subgraph with the question
        # The graphrag graph expects a GraphRAGQueryState with 'question' field
        result = await graphrag_graph.ainvoke(
            {"question": question}, config={"recursion_limit": 50}
        )

        # Extract the answer from the result
        # The graphrag agent returns messages in its state
        if result.get("messages"):
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                answer = last_message.content
            else:
                answer = str(last_message)
        else:
            answer = "No answer generated by GraphRAG agent"

        print(f"✅ GraphRAG Agent completed. Answer length: {len(answer)} chars")

    except Exception as e:
        answer = f"Error in GraphRAG agent: {e!s}"
        print(f"❌ GraphRAG Agent error: {e}")

    # Return the answer as an AI message
    return Command(
        goto=END,
        update={"messages": [AIMessage(content=answer)], "final_answer": answer},
    )


# --------------------------------------------------------------------------- #
# 6) Build Supervisor Graph
# --------------------------------------------------------------------------- #


def build_supervisor_graph() -> StateGraph:
    """Build and compile the supervisor graph with subgraphs."""
    builder = StateGraph(SupervisorState)

    # Add nodes
    builder.add_node("supervisor_decision", supervisor_decision)
    builder.add_node("cypher_agent_wrapper", cypher_agent_wrapper)
    builder.add_node("graphrag_agent_wrapper", graphrag_agent_wrapper)

    # Add edges
    builder.add_edge(START, "supervisor_decision")
    # The supervisor_decision node uses Command to route dynamically

    # Compile and return
    return builder.compile()


# Create the compiled supervisor graph
supervisor_graph = build_supervisor_graph()


# --------------------------------------------------------------------------- #
# 7) Main Execution (for testing)
# --------------------------------------------------------------------------- #

if __name__ == "__main__":

    async def test_supervisor():
        """Test the supervisor with different types of questions."""
        test_questions = [
            # Should go to cypher_agent
            "¿Cuántos proyectos hay en total?",
            "¿Qué comunas tienen proyectos?",
            "Lista todos los proyectos en la región de Antofagasta",
            # Should go to graphrag_agent
            "¿Qué especies de flora se encuentran en los proyectos?",
            "Resume el impacto ambiental de los proyectos",
            "¿Cuáles son las especies en peligro mencionadas?",
        ]

        # Test with one question
        test_question = test_questions[3]  # GraphRAG question about species

        print("=" * 70)
        print("🤖 SUPERVISOR AGENT TEST")
        print("=" * 70)
        print(f"Question: {test_question}")
        print("-" * 70)

        try:
            # Run the supervisor graph
            result = await supervisor_graph.ainvoke(
                {"question": test_question},
                config={"recursion_limit": 50},
                debug=False,  # Set to True for detailed execution trace
            )

            # Display the final answer
            if result.get("final_answer"):
                print("\n" + "=" * 70)
                print("📝 FINAL ANSWER:")
                print("-" * 70)
                print(result["final_answer"])
                print("=" * 70)
            else:
                print("\n⚠️ No final answer generated")

        except Exception as e:
            print(f"\n❌ Error running supervisor: {e}")
            import traceback

            traceback.print_exc()

    # Run the async test function
    asyncio.run(test_supervisor())
