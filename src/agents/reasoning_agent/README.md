# Reasoning Agent

## Overview

The Reasoning Agent is a specialized LangGraph-based agent designed to perform high-level intellectual tasks on data and results from other agents. It excels at summarizing, analyzing, reflecting, interpreting, and synthesizing complex information.

## Capabilities

The agent supports the following intellectual activities:
- **Summarize**: Create concise summaries of complex data
- **Describe**: Provide detailed descriptions
- **Reflect**: Think deeply about implications
- **Analyze**: Break down information into components
- **Think**: General reasoning and problem solving
- **Read**: Extract and understand information
- **Judge**: Make evaluative decisions
- **Interpret**: Explain meaning and significance
- **Synthesize**: Combine multiple elements
- **Compare**: Identify similarities and differences
- **Evaluate**: Assess quality or value

## Architecture

### Components

1. **State Management** (`schemas.py`)
   - `ReasoningState`: Manages instruction, results, and outputs
   - `ReasoningTask`: Parsed reasoning task with type and focus
   - `ReasoningResponse`: Structured reasoning output with confidence

2. **LLM Chains** (`llm_chains.py`)
   - `get_task_parser_chain`: Parses instructions into structured tasks
   - `get_reasoning_engine_chain`: Performs the reasoning
   - `get_synthesizer_chain`: Formats final output

3. **Agent Logic** (`agent_logic.py`)
   - `parse_instruction`: Understands the reasoning request
   - `reason`: Executes the reasoning task
   - `synthesize`: Generates the final output

4. **Graph Builder** (`graph_builder.py`)
   - Constructs the LangGraph workflow
   - Defines the three-node pipeline

## Usage

### Standalone Usage

```python
from src.agents.reasoning_agent import graph

result = await graph.ainvoke({
    "instruction": "Analyze the ecological significance of these findings",
    "current_results": [
        "979 species found",
        "50% endemic species"
    ],
    "partial_results": []
})

print(result["final_output"])
```

### Integration with ReAct Agent

The reasoning agent is automatically available as a tool in the ReAct agent:

```python
Step(
    instruction="Summarize and analyze the key findings",
    suggested_tool="Reasoning_agent",
    reasoning="Synthesize insights from previous results",
    result="",
    is_complete=False
)
```

## Input Format

```python
{
    "instruction": str,           # The reasoning task to perform
    "current_results": list[Any], # Complete results from previous steps
    "partial_results": list[Any]  # Partial or intermediate results
}
```

## Output Format

```python
{
    "final_output": str,  # The synthesized output
    "reasoning_response": {
        "reasoning": str,      # The reasoning process
        "conclusion": str,     # Final conclusion
        "confidence": float,   # Confidence level (0-1)
        "key_points": list[str] # Key insights
    }
}
```

## Flow Diagram

```
START
  |
  v
parse_instruction (Understand task)
  |
  v
reason (Execute reasoning)
  |
  v
synthesize (Format output)
  |
  v
END
```

## Testing

Run the tests:

```bash
# Standalone tests
uv run -m src.agents.reasoning_agent.graph_builder

# Integration with ReAct
uv run -m src.agents.ReAct_agent.test_reasoning_integration
```

## Examples

### Summarization
```python
instruction = "Summarize the key findings from these project results"
```

### Analysis
```python
instruction = "Analyze the ecological significance of the flora data"
```

### Synthesis
```python
instruction = "Synthesize insights from all previous steps and identify patterns"
```

## Key Features

1. **Flexible Task Recognition**: Automatically identifies the type of intellectual task
2. **Context-Aware**: Uses all previous results for informed reasoning
3. **Confidence Scoring**: Provides confidence levels for conclusions
4. **Key Point Extraction**: Identifies and highlights important insights
5. **Error Resilience**: Graceful fallbacks for parsing or reasoning errors
