"""Decision engine for retrieval gating and mode selection."""

from __future__ import annotations

import logging
from typing import Optional

from raglib.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)

RETRIEVAL_VECTOR: str = "vector"
RETRIEVAL_WEB: str = "web"
RETRIEVAL_HYBRID: str = "hybrid"
RETRIEVAL_NONE: str = "none"
ALLOWED_RETRIEVAL_TYPES = {
    RETRIEVAL_VECTOR,
    RETRIEVAL_WEB,
    RETRIEVAL_HYBRID,
    RETRIEVAL_NONE,
}


class DecisionEngine:
    """Decides whether retrieval is needed and what retrieval mode to use."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        """Initialize decision policy with an optional LLM client."""

        self.llm_client = llm_client

    def should_retrieve(self, query: str) -> bool:
        """Return whether retrieval should run for this query."""

        route = self.retrieval_type(query)
        decision = route != RETRIEVAL_NONE
        logger.info("DecisionEngine should_retrieve=%s route=%s", decision, route)
        return decision

    def retrieval_type(self, query: str) -> str:
        """Return one of: vector, web, hybrid, none."""

        if self.llm_client is not None:
            prompt = (
                "Decide retrieval_type for this query. Allowed: vector, web, hybrid, none.\n"
                f"Query: {query}\n"
                "Return one token only."
            )
            response = self.llm_client.complete(prompt=prompt).strip().lower()
            if response in ALLOWED_RETRIEVAL_TYPES:
                logger.info("DecisionEngine selected via LLM: %s", response)
                return response

        lower = query.lower().strip()
        if not lower:
            return RETRIEVAL_NONE
        if any(token in lower for token in ("hello", "thanks", "hi", "good morning")):
            return RETRIEVAL_NONE
        if any(token in lower for token in ("latest", "today", "news", "current", "web")):
            return RETRIEVAL_WEB
        if any(token in lower for token in ("compare", "versus", "vs", "trend")):
            return RETRIEVAL_HYBRID
        return RETRIEVAL_VECTOR
