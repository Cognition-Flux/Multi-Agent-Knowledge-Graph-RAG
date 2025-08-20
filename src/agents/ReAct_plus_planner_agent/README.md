# ReAct + Planner Combined Agent

## Overview

The ReAct + Planner agent is a complete workflow that combines intelligent planning with robust execution. It ensures tool awareness throughout the process to prevent hallucinations and provides a seamless question-to-answer pipeline.

## Key Features

### 🎯 Tool Awareness
- **Prevents Hallucinations**: Both planner and executor are explicitly aware of available tools
- **Tool Registry**: Centralized registry with descriptions and use cases
- **Validation**: Automatic validation and correction of tool names in plans

### 📋 Intelligent Planning
- **Context-Aware**: Plans are generated with full knowledge of available tools
- **Direct Responses**: Can provide immediate answers for simple questions
- **Multi-Step Reasoning**: Creates complex plans for sophisticated queries

### 🚀 Robust Execution
- **ReAct Pattern**: Combines reasoning and acting for reliable execution
- **Error Handling**: Graceful recovery from tool failures
- **Progress Tracking**: Real-time monitoring of execution progress

## Architecture

### Workflow Stages

```
START
  |
  v
inject_tools_info (Inject available tools information)
  |
  v
generate_plan (Create plan with tool awareness)
  |
  v
  ├─> direct_answer (If simple question)
  │     |
  │     v
  │   finalize
  │
  └─> validate_plan (Ensure tools are valid)
        |
        v
      execute_with_react (Run the plan)
        |
        v
      finalize (Prepare final answer)
        |
        v
      END
```

### Available Tools

1. **cypher_query_agent**
   - Executes Cypher queries on Neo4j
   - For metadata, counts, filtering
   - Keywords: cypher, metadata, query

2. **hybrid_graphRAG_agent**
   - Retrieves document content
   - For text search, chunk analysis
   - Keywords: hybrid, graphrag, content

3. **reasoning_agent**
   - Performs intellectual tasks
   - For analysis, synthesis, summarization
   - Keywords: reasoning, analyze, summarize

## Components

### `tools_registry.py`
Maintains the registry of available tools with descriptions and use cases.

### `schemas.py`
Defines the combined state that tracks both planning and execution.

### `agent_logic.py`
Contains the workflow nodes:
- `inject_tools_info`: Adds tool information to context
- `generate_plan`: Creates an execution plan
- `validate_plan`: Validates tool names
- `execute_with_react`: Runs the plan
- `direct_answer`: Handles simple responses
- `finalize`: Prepares the final answer

### `graph_builder.py`
Assembles the complete workflow graph.

## Usage

### Basic Usage

```python
from src.agents.ReAct_plus_planner_agent import graph

# Ask a question
result = await graph.ainvoke({
    "question": "What projects exist in Antofagasta region?"
})

print(result["final_answer"])
```

### Streaming Execution

```python
# Stream updates to see progress
async for chunk in graph.astream(
    {"question": "Analyze flora in the region"},
    stream_mode="updates"
):
    print(f"Update from {list(chunk.keys())[0]}")
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
uv run -m src.agents.ReAct_plus_planner_agent.graph_builder
```

The test suite includes:
1. **Metadata Query Test**: Tests cypher_query_agent usage
2. **Complex Multi-Tool Test**: Tests orchestration of multiple tools
3. **Direct Response Test**: Tests simple question handling
4. **Streaming Test**: Tests real-time progress monitoring

## Error Handling

The agent handles various error scenarios:
- **Invalid Tools**: Automatically maps to valid alternatives
- **Execution Failures**: Continues with remaining steps
- **Planning Errors**: Falls back to safe responses

## Tool Validation

The system validates tools at multiple levels:
1. **Planning Phase**: Tools are suggested from known registry
2. **Validation Phase**: Tool names are verified and corrected
3. **Execution Phase**: Final fallback to reasoning_agent

## Examples

### Simple Metadata Query
```python
question = "How many projects are in region X?"
# Will use: cypher_query_agent
```

### Content Search
```python
question = "Find information about flora species"
# Will use: hybrid_graphRAG_agent
```

### Complex Analysis
```python
question = "Summarize projects and analyze their environmental impact"
# Will use: cypher_query_agent → hybrid_graphRAG_agent → reasoning_agent
```

## Benefits

1. **No Hallucinations**: Tools are explicitly defined and validated
2. **Intelligent Routing**: Automatically selects appropriate tools
3. **Complete Pipeline**: From question to final answer
4. **Observable**: Full visibility into planning and execution
5. **Resilient**: Handles errors gracefully
6. **Extensible**: Easy to add new tools to registry
