"""Web retrieval component backed by provider abstraction."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import List

from raglib.providers.base_provider import BaseSearchProvider, ProviderError
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class WebRetriever:
    """Retrieves documents from a web search provider."""

    def __init__(self, provider: BaseSearchProvider, top_k: int = 5):
        """Initialize the web retriever with a provider."""

        self.provider = provider
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve top-k web documents for a query."""

        logger.info("WebRetriever querying provider=%s", self.provider.name)
        try:
            docs = self.provider.search(query=query, num_results=self.top_k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("WebRetriever provider failed: %s", exc)
            raise ProviderError(f"Web retrieval failed for provider={self.provider.name}: {exc}") from exc
        return [
            replace(doc, source=doc.source or self.provider.name)
            for doc in docs[: self.top_k]
        ]
