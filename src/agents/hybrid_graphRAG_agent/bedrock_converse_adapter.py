"""Adapter to use LangChain ChatBedrockConverse with neo4j_graphrag.

This wraps a LangChain `ChatBedrockConverse` instance to conform to
`neo4j_graphrag.llm.base.LLMInterface` so GraphRAG can pass
`system_instruction` and optional `message_history` without surfacing
unsupported kwargs to the Converse API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.types import LLMMessage


class BedrockConverseLLMAdapter(LLMInterface):
    """Bridge between LangChain ChatBedrockConverse and GraphRAG LLMInterface.

    Parameters
    ----------
    chat
        A configured LangChain `ChatBedrockConverse` instance.
    model_name
        Logical model name for GraphRAG visibility; defaults to chat.model_id.
    """

    def __init__(self, chat: Any, model_name: str | None = None):
        # model_params are not required here; they live inside the wrapped chat
        super().__init__(model_name=model_name or getattr(chat, "model_id", "bedrock"))
        self.chat = chat

    def _convert_history(
        self, message_history: list[LLMMessage] | MessageHistory | None
    ) -> list[tuple[str, str]]:
        if message_history is None:
            return []
        if isinstance(message_history, MessageHistory):
            history = message_history.messages
        else:
            history = message_history
        converted: list[tuple[str, str]] = []
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("system", "user", "assistant"):
                if role == "system":
                    lc_role = "system"
                elif role == "user":
                    lc_role = "human"
                else:
                    lc_role = "ai"
                converted.append((lc_role, content))
        return converted

    def _build_messages(
        self,
        user_text: str,
        message_history: list[LLMMessage] | MessageHistory | None = None,
        system_instruction: str | None = None,
    ) -> Sequence[BaseMessage | tuple[str, str]]:
        messages: list[tuple[str, str]] = []
        if system_instruction:
            messages.append(("system", system_instruction))
        messages.extend(self._convert_history(message_history))
        messages.append(("human", user_text))
        return messages

    def invoke(
        self,
        user_text: str,
        message_history: list[LLMMessage] | MessageHistory | None = None,
        system_instruction: str | None = None,
    ) -> LLMResponse:
        """Invoke the underlying Bedrock Converse chat model."""
        messages = self._build_messages(
            user_text=user_text,
            message_history=message_history,
            system_instruction=system_instruction,
        )
        ai_msg: AIMessage = self.chat.invoke(messages)  # type: ignore[assignment]
        if isinstance(ai_msg.content, str):
            content = ai_msg.content
        else:
            content = "".join(
                block.get("text", "")
                for block in (ai_msg.content or [])  # type: ignore[index]
            )
        return LLMResponse(content=content)

    async def ainvoke(
        self,
        user_text: str,
        message_history: list[LLMMessage] | MessageHistory | None = None,
        system_instruction: str | None = None,
    ) -> LLMResponse:
        """Async invoke for completeness; GraphRAG currently uses sync path."""
        messages = self._build_messages(
            user_text=user_text,
            message_history=message_history,
            system_instruction=system_instruction,
        )
        ai_msg: AIMessage = await self.chat.ainvoke(messages)  # type: ignore[attr-defined]
        if isinstance(ai_msg.content, str):
            content = ai_msg.content
        else:
            content = "".join(
                block.get("text", "")
                for block in (ai_msg.content or [])  # type: ignore[index]
            )
        return LLMResponse(content=content)
