"""Registry of available tools for the planner and ReAct agents.

This module maintains a registry of all available tools with their
descriptions to ensure both planner and ReAct agents are aware of
what tools exist and how to use them.
"""

from typing import TypedDict


class ToolInfo(TypedDict):
    """Information about a tool."""

    name: str
    description: str
    use_cases: list[str]
    keywords: list[str]


# Registry of all available tools
AVAILABLE_TOOLS: list[ToolInfo] = [
    {
        "name": "cypher_query_agent",
        "description": "Executes Cypher queries on Neo4j knowledge graph to retrieve metadata and structured information",
        "use_cases": [
            "Query project metadata",
            "Find entities by attributes",
            "Retrieve relationships between entities",
            "Count and aggregate data",
            "Filter by region, comuna, or other metadata",
        ],
        "keywords": [
            "cypher",
            "metadata",
            "neo4j",
            "graph",
            "query",
            "filter",
            "count",
        ],
    },
    {
        "name": "hybrid_graphRAG_agent",
        "description": "Performs hybrid graph RAG queries to retrieve and analyze document content and chunks",
        "use_cases": [
            "Search document content",
            "Find information in text chunks",
            "Retrieve flora and fauna descriptions",
            "Access environmental impact studies",
            "Get detailed project descriptions",
        ],
        "keywords": [
            "hybrid",
            "graphrag",
            "content",
            "chunk",
            "document",
            "text",
            "search",
        ],
    },
    {
        "name": "reasoning_agent",
        "description": "Performs intellectual tasks like summarizing, analyzing, reflecting, and synthesizing information",
        "use_cases": [
            "Summarize findings",
            "Analyze patterns",
            "Synthesize information from multiple sources",
            "Interpret results",
            "Reflect on implications",
            "Compare and evaluate data",
        ],
        "keywords": [
            "reasoning",
            "reason",
            "think",
            "analyze",
            "summarize",
            "synthesize",
            "interpret",
            "reflect",
        ],
    },
]


def get_tools_description() -> str:
    """Get a formatted description of all available tools."""
    descriptions = []
    for tool in AVAILABLE_TOOLS:
        tool_desc = f"- **{tool['name']}**: {tool['description']}\n"
        tool_desc += f"  Use cases: {', '.join(tool['use_cases'][:3])}\n"
        tool_desc += f"  Keywords: {', '.join(tool['keywords'][:4])}"
        descriptions.append(tool_desc)

    return "\n\n".join(descriptions)


def get_tool_by_name(name: str) -> ToolInfo | None:
    """Get tool information by name."""
    for tool in AVAILABLE_TOOLS:
        if tool["name"] == name:
            return tool
    return None


def suggest_tool(description: str) -> str:
    """Suggest the most appropriate tool based on a task description."""
    description_lower = description.lower()

    # Check keywords for each tool
    scores = {}
    for tool in AVAILABLE_TOOLS:
        score = 0
        for keyword in tool["keywords"]:
            if keyword in description_lower:
                score += 1
        scores[tool["name"]] = score

    # Return the tool with highest score, default to reasoning_agent
    if max(scores.values()) > 0:
        return max(scores, key=scores.get)
    return "reasoning_agent"  # Default for intellectual tasks


def format_tools_for_prompt() -> str:
    """Format tools information for inclusion in prompts."""
    output = "## Available Tools\n\n"
    output += "You have access to the following tools:\n\n"
    output += get_tools_description()
    output += "\n\n**Important**: Only use these exact tool names. Do not hallucinate or invent tool names."
    return output
