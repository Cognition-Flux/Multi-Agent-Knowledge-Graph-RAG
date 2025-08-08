"""LangGraph (Python) — pasar valores en runtime a tools de forma simple.

- `user_id` se inyecta en runtime desde `config.configurable` (no visible al LLM).
- Se usa un closure para leer el estado actual (primer mensaje humano) y guardarlo.

Guía base (JS): https://langchain-ai.github.io/langgraphjs/how-tos/pass-run-time-values-to-tools/
"""

# %%

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from langgraph.graph import END, START, MessagesState, StateGraph

from src.utils import get_llm


# Almacenamiento en memoria (simula persistencia por usuario)
_user_store: dict[str, dict[str, object]] = {}


def generate_tools(state: MessagesState):
    """Crea tools que cierran sobre el `state` (p. ej. primer mensaje humano)."""
    # Primer mensaje humano del hilo actual (contexto dinámico)
    try:
        initial_user_content = next(
            (m.content for m in state["messages"] if isinstance(m, HumanMessage)),
            "",
        )
    except Exception:
        initial_user_content = ""

    @tool
    def update_favorite_pets(
        pets: list[str], user_id: Annotated[str, InjectedToolArg]
    ) -> str:
        """Agrega `pets` para `user_id` y guarda el contexto del primer mensaje."""
        _user_store[user_id] = {
            "pets": list(pets),
            "context": initial_user_content,
        }
        return "update_favorite_pets llamado."

    @tool
    def get_favorite_pets(user_id: Annotated[str, InjectedToolArg]) -> str:
        """Devuelve `{"pets": [...], "context": "..."}` para `user_id`."""
        data = _user_store.get(user_id) or {}
        pets = data.get("pets", [])
        context = data.get("context", "")
        return str({"pets": pets, "context": context})

    return [get_favorite_pets, update_favorite_pets]


def route_after_agent(state: MessagesState):
    """Si el LLM pide tools → "tools"; si no → END."""
    if not state["messages"]:
        return END
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return END


def call_agent(state: MessagesState) -> dict:
    """El LLM decide si usar tools (schema visible solo para `pets`)."""
    llm = get_llm()
    tools = generate_tools(state)
    model_with_tools = llm.bind_tools(tools)

    system_prompt = "Eres un asistente personal. Guarda preferencias del usuario."

    response = model_with_tools.invoke(
        [
            {"role": "system", "content": system_prompt},
            *state["messages"],
        ]
    )
    return {"messages": [response]}


def call_tools(state: MessagesState, config: RunnableConfig) -> dict:
    """Ejecuta tools del último AIMessage y añade `user_id` desde config."""
    tools = generate_tools(state)
    tool_map = {t.name: t for t in tools}

    last = state["messages"][-1]
    if not (isinstance(last, AIMessage) and getattr(last, "tool_calls", None)):
        return {}

    tool_messages: list[BaseMessage] = []
    for tc in last.tool_calls:
        name = tc.get("name")
        args = tc.get("args", {})
        tool_id = tc.get("id")
        tool_fn = tool_map.get(name)
        if tool_fn is None:
            continue
        # Inyección explícita del user_id desde config.configurable si falta
        if isinstance(config, dict):
            cfg_user = (
                config.get("configurable", {}).get("user_id")  # type: ignore[assignment]
                if config.get("configurable")
                else None
            )
            if cfg_user is not None and "user_id" not in args:
                args["user_id"] = cfg_user
        # Importante: pasamos el `config` para permitir compat con InjectedToolArg
        result = tool_fn.invoke(args, config=config)
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_id, name=name)
        )

    return {"messages": tool_messages}


# Graph minimalista (agent → tools → agent/END)
builder = StateGraph(MessagesState)
builder.add_node("agent", call_agent)
builder.add_node("tools", call_tools)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent)
builder.add_edge("tools", "agent")

graph = builder.compile(debug=True)


def _print_messages(messages: list[BaseMessage]) -> None:
    """Salida legible para consola (similar a la guía)."""
    for m in messages:
        if isinstance(m, HumanMessage):
            print(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            if m.content:
                print(f"Assistant: {m.content}")
            if getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    print(f"Tool call: {tc.get('name')}({tc.get('args')!s})")
        elif isinstance(m, ToolMessage):
            print(f"{m.name} tool output: {m.content}")


if __name__ == "__main__":
    # 1) Guardar preferencias (closure + `user_id` por config)
    inputs = {
        "messages": [
            HumanMessage(
                content=(
                    "Mi mascota favorita es un terrier. Vi uno muy lindo en Twitter."
                )
            )
        ]
    }
    config = {
        "configurable": {
            "thread_id": "1",
            "user_id": "a-user",
        }
    }

    result = graph.invoke(inputs, config)
    _print_messages(
        result["messages"]
    )  # Debería mostrar la llamada a update_favorite_pets

    # 2) Recuperar preferencias (nuevo thread, mismo user)
    inputs = {
        "messages": [
            HumanMessage(
                content=(
                    "¿Cuáles son mis mascotas favoritas y qué dije cuando te las conté?"
                )
            )
        ]
    }
    config = {
        "configurable": {
            "thread_id": "2",
            "user_id": "a-user",
        }
    }

    result = graph.invoke(inputs, config)
    _print_messages(
        result["messages"]
    )  # Debería mostrar la llamada a get_favorite_pets
