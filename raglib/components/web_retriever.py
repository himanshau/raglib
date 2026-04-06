"""Web retrieval component backed by provider abstraction."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional

from raglib.providers import (
    BingProvider,
    BraveProvider,
    DuckDuckGoProvider,
    ExaProvider,
    GoogleCSEProvider,
    SearxNGProvider,
    SerpAPIProvider,
    TavilyProvider,
)
from raglib.providers.base_provider import (
    BaseSearchProvider,
    ProviderError,
    ProviderNotConfiguredError,
)
from raglib.schemas import Document

logger = logging.getLogger(__name__)


WEB_PROVIDER_ALIASES: Dict[str, str] = {
    "ddg": "duckduckgo",
    "duckduckgo_search": "duckduckgo",
    "google": "google_cse",
}

WEB_PROVIDERS_REQUIRING_API_KEY = {
    "tavily",
    "serpapi",
    "brave",
    "bing",
    "google_cse",
    "exa",
    "searxng",
}


class WebRetriever:
    """Retrieves documents from web providers with configurable fallback."""

    def __init__(
        self,
        provider: Optional[BaseSearchProvider] = None,
        provider_name: str = "duckduckgo",
        provider_api_key: Optional[str] = None,
        provider_kwargs: Optional[Dict[str, Any]] = None,
        fallback_provider: Optional[BaseSearchProvider] = None,
        enable_duckduckgo_fallback: bool = True,
        top_k: int = 5,
        fail_silently: bool = True,
    ):
        """Initialize web retriever using provider instance or provider config."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if provider is not None:
            self.provider = provider
        else:
            self.provider = self._build_provider(
                provider_name=provider_name,
                provider_api_key=provider_api_key,
                provider_kwargs=provider_kwargs,
            )

        self.top_k = top_k
        self.fail_silently = fail_silently

        if fallback_provider is not None:
            self.fallback_provider = fallback_provider
        elif enable_duckduckgo_fallback and self.provider.name != DuckDuckGoProvider.PROVIDER_NAME:
            self.fallback_provider = DuckDuckGoProvider()
        else:
            self.fallback_provider = None

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve top-k web documents for a query."""

        if not isinstance(query, str) or not query.strip():
            return []

        logger.info("WebRetriever querying provider=%s", self.provider.name)
        try:
            docs = self.provider.search(query=query, num_results=self.top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("WebRetriever provider failed: %s", exc)
            fallback_docs = self._retrieve_from_fallback(query=query, cause=exc)
            if fallback_docs:
                return fallback_docs
            if self.fail_silently:
                return []
            raise ProviderError(f"Web retrieval failed for provider={self.provider.name}: {exc}") from exc

        return self._normalize_documents(docs=docs, provider_name=self.provider.name)

    def _retrieve_from_fallback(self, query: str, cause: Exception) -> List[Document]:
        """Retrieve from fallback provider after primary provider failure."""

        if self.fallback_provider is None:
            return []

        logger.warning(
            "WebRetriever falling back from provider=%s to provider=%s due to: %s",
            self.provider.name,
            self.fallback_provider.name,
            cause,
        )

        try:
            docs = self.fallback_provider.search(query=query, num_results=self.top_k)
        except Exception as fallback_exc:  # noqa: BLE001
            logger.warning("WebRetriever fallback provider failed: %s", fallback_exc)
            if self.fail_silently:
                return []
            raise ProviderError(
                "Web retrieval failed for primary provider "
                f"'{self.provider.name}' and fallback provider '{self.fallback_provider.name}'"
            ) from fallback_exc

        return self._normalize_documents(docs=docs, provider_name=self.fallback_provider.name)

    def _normalize_documents(self, docs: List[Document], provider_name: str) -> List[Document]:
        """Normalize documents to unified title/content/url/score metadata schema."""

        normalized: List[Document] = []
        for idx, doc in enumerate(docs[: self.top_k]):
            title = str(doc.metadata.get("title", "")).strip() or self._derive_title(doc.content)
            url = str(doc.metadata.get("url") or doc.metadata.get("link") or "").strip()
            snippet = str(doc.metadata.get("snippet", "")).strip() or doc.content

            score = float(doc.score)
            if score == 0.0:
                score = max(0.0, 1.0 - (idx / max(self.top_k, 1)))

            metadata = {
                **doc.metadata,
                "title": title,
                "snippet": snippet,
                "url": url,
                "provider": doc.metadata.get("provider", provider_name),
                "score": score,
            }

            normalized.append(
                replace(
                    doc,
                    content=snippet,
                    score=score,
                    source=doc.source or provider_name,
                    metadata=metadata,
                )
            )

        return normalized

    def _build_provider(
        self,
        provider_name: str,
        provider_api_key: Optional[str],
        provider_kwargs: Optional[Dict[str, Any]],
    ) -> BaseSearchProvider:
        """Build provider instance from name and config for plug-and-play setup."""

        kwargs = dict(provider_kwargs or {})
        normalized_name = self._normalize_provider_name(provider_name)

        if normalized_name in WEB_PROVIDERS_REQUIRING_API_KEY and not provider_api_key:
            raise ProviderNotConfiguredError(
                f"web provider '{normalized_name}' requires a valid API key"
            )

        if normalized_name == "duckduckgo":
            return DuckDuckGoProvider(
                region=str(kwargs.pop("region", "wt-wt")),
                safesearch=str(kwargs.pop("safesearch", "moderate")),
                timelimit=kwargs.pop("timelimit", None),
                backend=str(kwargs.pop("backend", "auto")),
            )

        if normalized_name == "tavily":
            return TavilyProvider(
                api_key=provider_api_key,
                topic=str(kwargs.pop("topic", "general")),
                search_depth=str(kwargs.pop("search_depth", "basic")),
                include_answer=bool(kwargs.pop("include_answer", False)),
                timeout=int(kwargs.pop("timeout", 10)),
            )

        if normalized_name == "serpapi":
            return SerpAPIProvider(
                api_key=provider_api_key,
                engine=str(kwargs.pop("engine", "google")),
                location=str(kwargs.pop("location", "United States")),
                timeout=int(kwargs.pop("timeout", 10)),
            )

        if normalized_name == "brave":
            return BraveProvider(
                api_key=provider_api_key,
                country=str(kwargs.pop("country", "us")),
                search_lang=str(kwargs.pop("search_lang", "en")),
                timeout=int(kwargs.pop("timeout", 10)),
            )

        if normalized_name == "bing":
            return BingProvider(
                api_key=provider_api_key,
                market=str(kwargs.pop("market", "en-US")),
                timeout=int(kwargs.pop("timeout", 10)),
            )

        if normalized_name == "google_cse":
            cse_id = str(kwargs.pop("cse_id", "")).strip()
            if not cse_id:
                raise ProviderNotConfiguredError(
                    "web provider 'google_cse' requires cse_id in provider_kwargs"
                )
            return GoogleCSEProvider(
                api_key=provider_api_key,
                cse_id=cse_id,
                safe=str(kwargs.pop("safe", "off")),
                timeout=int(kwargs.pop("timeout", 10)),
            )

        if normalized_name == "exa":
            return ExaProvider(
                api_key=provider_api_key,
                use_autoprompt=bool(kwargs.pop("use_autoprompt", True)),
                timeout=int(kwargs.pop("timeout", 10)),
            )

        if normalized_name == "searxng":
            base_url = str(kwargs.pop("base_url", "")).strip()
            if not base_url:
                raise ProviderNotConfiguredError(
                    "web provider 'searxng' requires base_url in provider_kwargs"
                )
            return SearxNGProvider(
                api_key=provider_api_key,
                base_url=base_url,
                categories=str(kwargs.pop("categories", "general")),
                language=str(kwargs.pop("language", "en")),
                timeout=int(kwargs.pop("timeout", 10)),
            )

        raise ValueError(f"Unsupported web provider '{provider_name}'")

    def _normalize_provider_name(self, provider_name: str) -> str:
        """Normalize provider aliases to canonical names."""

        normalized = (provider_name or DuckDuckGoProvider.PROVIDER_NAME).strip().lower()
        return WEB_PROVIDER_ALIASES.get(normalized, normalized)

    def _derive_title(self, content: str) -> str:
        """Derive a compact title when provider response omits one."""

        line = content.strip().split("\n", maxsplit=1)[0].strip()
        if not line:
            return "Web result"
        return line[:120]
