"""Query router that dispatches to vector, web, or hybrid retrieval."""

from __future__ import annotations

import logging
from typing import List, Optional

from raglib.components.hybrid_retriever import HybridRetriever
from raglib.components.retriever import Retriever
from raglib.components.web_retriever import WebRetriever
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document

logger = logging.getLogger(__name__)

ROUTE_VECTOR: str = "vector"
ROUTE_WEB: str = "web"
ROUTE_HYBRID: str = "hybrid"


class RouterRetriever:
    """Routes retrieval requests to the most suitable retriever."""

    def __init__(
        self,
        vector_retriever: Retriever,
        web_retriever: WebRetriever,
        hybrid_retriever: Optional[HybridRetriever] = None,
        llm_client: Optional[BaseLLMClient] = None,
    ):
        """Initialize routing dependencies and optional LLM policy."""

        self.vector_retriever = vector_retriever
        self.web_retriever = web_retriever
        self.hybrid_retriever = hybrid_retriever
        self.llm_client = llm_client

    def route(self, query: str) -> str:
        """Return the best retrieval route for the query."""

        lower = query.lower()
        if self.llm_client is not None:
            prompt = (
                "Choose one retrieval mode: vector, web, hybrid.\n"
                f"Query: {query}"
            )
            candidate = self.llm_client.complete(prompt=prompt).strip().lower()
            if candidate in {ROUTE_VECTOR, ROUTE_WEB, ROUTE_HYBRID}:
                logger.info("RouterRetriever route selected by LLM: %s", candidate)
                return candidate

        if any(token in lower for token in ("today", "latest", "news", "current", "web")):
            route = ROUTE_WEB
        elif any(token in lower for token in ("compare", "trend", "benchmark", "vs")):
            route = ROUTE_HYBRID
        else:
            route = ROUTE_VECTOR

        logger.info("RouterRetriever route selected by heuristic: %s", route)
        return route

    def retrieve(self, query: str) -> List[Document]:
        """Retrieve documents based on selected route."""

        route = self.route(query)
        if route == ROUTE_WEB:
            return self.web_retriever.retrieve(query)
        if route == ROUTE_HYBRID and self.hybrid_retriever is not None:
            return self.hybrid_retriever.retrieve(query)
        return self.vector_retriever.retrieve(query)
