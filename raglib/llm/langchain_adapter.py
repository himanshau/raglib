"""LangChain chat model adapter implementing raglib BaseLLMClient."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from raglib.llm.base_client import BaseLLMClient
from raglib.llm.constants import OLLAMA_DEFAULT_BASE_URL, PROVIDER_DEFAULT_CHAT_MODEL

logger = logging.getLogger(__name__)


class LangChainAdapter(BaseLLMClient):
    """Bridge a LangChain chat model into raglib's BaseLLMClient interface."""

    def __init__(self, langchain_model: Any, provider_name: str = "unknown"):
        """Store the injected LangChain chat model and provider metadata."""

        self._model = langchain_model
        self._provider = provider_name
        self._logger = logging.getLogger(__name__)

    def complete(self, prompt: str, system: str = "") -> str:
        """Execute a single prompt completion using LangChain message objects."""

        payload: Any
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages: List[Any] = []
            if system:
                messages.append(SystemMessage(content=system))
            messages.append(HumanMessage(content=prompt))
            payload = messages
        except ImportError:
            logger.warning(
                "langchain-core is not installed; using plain-text invoke fallback for provider='%s'",
                self._provider,
            )
            payload = f"{system}\n\n{prompt}".strip() if system else prompt

        self._logger.debug("[%s] Sending completion request.", self._provider)
        response = self._model.invoke(payload)
        return self._coerce_content(getattr(response, "content", response))

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Execute a multi-turn chat completion from role/content dictionaries."""

        lc_messages: List[Any] = []
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

            role_map = {
                "user": HumanMessage,
                "human": HumanMessage,
                "assistant": AIMessage,
                "ai": AIMessage,
                "system": SystemMessage,
            }

            for message in messages:
                role = message.get("role", "user").lower()
                msg_cls = role_map.get(role, HumanMessage)
                lc_messages.append(msg_cls(content=message.get("content", "")))
            payload: Any = lc_messages
        except ImportError:
            logger.warning(
                "langchain-core is not installed; using plain-text chat fallback for provider='%s'",
                self._provider,
            )
            payload = "\n".join(
                f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages
            )

        self._logger.debug(
            "[%s] Sending chat request (%d messages).",
            self._provider,
            len(lc_messages),
        )
        response = self._model.invoke(payload)
        return self._coerce_content(getattr(response, "content", response))

    @classmethod
    def build(
        cls,
        provider: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ) -> "LangChainAdapter":
        """Build a provider-specific LangChain chat model and wrap it with this adapter."""

        normalized_provider = provider.strip().lower()
        resolved_model = model_name or PROVIDER_DEFAULT_CHAT_MODEL.get(normalized_provider)

        if not resolved_model:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                "Supported: openai, anthropic, groq, google, ollama"
            )

        if normalized_provider == "openai":
            try:
                from langchain_openai import ChatOpenAI
            except ImportError as exc:
                raise ImportError("pip install langchain-openai") from exc
            lc_model = ChatOpenAI(
                model=resolved_model,
                api_key=api_key,
                temperature=temperature,
            )
        elif normalized_provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError as exc:
                raise ImportError("pip install langchain-anthropic") from exc
            lc_model = ChatAnthropic(
                model=resolved_model,
                api_key=api_key,
                temperature=temperature,
            )
        elif normalized_provider == "groq":
            try:
                from langchain_groq import ChatGroq
            except ImportError as exc:
                raise ImportError("pip install langchain-groq") from exc
            lc_model = ChatGroq(
                model=resolved_model,
                api_key=api_key,
                temperature=temperature,
            )
        elif normalized_provider == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:
                raise ImportError("pip install langchain-google-genai") from exc
            lc_model = ChatGoogleGenerativeAI(
                model=resolved_model,
                google_api_key=api_key,
                temperature=temperature,
            )
        elif normalized_provider == "ollama":
            try:
                from langchain_ollama import ChatOllama
            except ImportError as exc:
                raise ImportError("pip install langchain-ollama") from exc
            resolved_base_url = base_url or OLLAMA_DEFAULT_BASE_URL
            lc_model = ChatOllama(
                model=resolved_model,
                base_url=resolved_base_url,
                temperature=temperature,
            )
        else:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                "Supported: openai, anthropic, groq, google, ollama"
            )

        logger.info(
            "Built LangChainAdapter for provider='%s' model='%s'",
            normalized_provider,
            resolved_model,
        )
        return cls(langchain_model=lc_model, provider_name=normalized_provider)

    @classmethod
    def _coerce_content(cls, content: Any) -> str:
        """Convert varying LangChain response content payloads into plain text."""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if text is not None:
                        parts.append(str(text))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()

        if isinstance(content, dict):
            text = content.get("text") or content.get("content")
            if text is not None:
                return str(text)

        return str(content)
