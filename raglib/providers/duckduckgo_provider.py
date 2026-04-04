"""DuckDuckGo free web provider implementation."""

from __future__ import annotations

import logging
from typing import List, Optional

from raglib.providers.base_provider import BaseSearchProvider, ProviderError
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class DuckDuckGoProvider(BaseSearchProvider):
    """Performs web search using duckduckgo-search."""

    PROVIDER_NAME: str = "duckduckgo"

    def __init__(
        self,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        backend: str = "api",
    ):
        """Initialize DuckDuckGo provider configuration."""

        self.region = region
        self.safesearch = safesearch
        self.timelimit = timelimit
        self.backend = backend

    @property
    def name(self) -> str:
        """Return the provider identifier."""

        return self.PROVIDER_NAME

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Execute DuckDuckGo search and map results to Document objects."""

        try:
            from duckduckgo_search import DDGS
        except ImportError as exc:
            message = (
                "duckduckgo-search is not installed. Install it with: "
                "pip install duckduckgo-search"
            )
            logger.error(message)
            raise ProviderError(message) from exc

        logger.info("Running DuckDuckGo search for query=%s", query)
        documents: List[Document] = []

        try:
            with DDGS() as ddgs:
                rows = ddgs.text(
                    query,
                    region=self.region,
                    safesearch=self.safesearch,
                    timelimit=self.timelimit,
                    backend=self.backend,
                    max_results=num_results,
                )
                for idx, row in enumerate((rows or [])[:num_results]):
                    title = str(row.get("title", ""))
                    snippet = str(row.get("body", ""))
                    url = str(row.get("href", ""))
                    content = f"{title}\n{snippet}".strip()
                    doc_id = f"ddg-{idx}-{abs(hash(url or content))}"
                    documents.append(
                        Document(
                            id=doc_id,
                            content=content,
                            metadata={"title": title, "url": url, "provider": self.name},
                            source=self.name,
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            logger.exception("DuckDuckGo search failed")
            raise ProviderError(f"DuckDuckGo search failed: {exc}") from exc

        return documents
