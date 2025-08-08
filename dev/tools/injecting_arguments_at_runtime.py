# %%

from copy import deepcopy
from typing import Annotated

from langchain_core.runnables import chain
from langchain_core.tools import InjectedToolArg, tool

from src.utils import get_llm


llm = get_llm()

user_to_pets = {}


@tool(parse_docstring=True)
def update_favorite_pets(
    pets: list[str], user_id: Annotated[str, InjectedToolArg]
) -> None:
    """Add the list of favorite pets.

    Args:
        pets: List of favorite pets to set.
        user_id: User's ID.
    """
    user_to_pets[user_id] = pets


@tool(parse_docstring=True)
def delete_favorite_pets(user_id: Annotated[str, InjectedToolArg]) -> None:
    """Delete the list of favorite pets.

    Args:
        user_id: User's ID.
    """
    user_to_pets.pop(user_id, None)


@tool(parse_docstring=True)
def list_favorite_pets(user_id: Annotated[str, InjectedToolArg]) -> None:
    """List favorite pets if any.

    Args:
        user_id: User's ID.
    """
    return user_to_pets.get(user_id, [])


tools = [
    update_favorite_pets,
    delete_favorite_pets,
    list_favorite_pets,
]
llm_with_tools = llm.bind_tools(tools)
ai_msg = llm_with_tools.invoke("my favorite animals are cats and parrots")
print(ai_msg.tool_calls)

USER_ID = "123"


@chain
def inject_user_id(ai_msg):
    tool_calls = []
    for tool_call in ai_msg.tool_calls:
        tool_call_copy = deepcopy(tool_call)
        tool_call_copy["args"]["user_id"] = USER_ID
        tool_calls.append(tool_call_copy)
    return tool_calls


tool_map = {tool.name: tool for tool in tools}


@chain
def tool_router(tool_call):
    return tool_map[tool_call["name"]]


chain = llm_with_tools | inject_user_id | tool_router.map()
chain.invoke("my favorite animals are cats and parrots")
print(user_to_pets)
# %%
