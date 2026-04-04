"""Pre-retrieval query rewriting component."""

from __future__ import annotations

import logging
from typing import Optional

from raglib.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Rewrites incoming queries into retrieval-friendly forms."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        """Initialize an optional LLM-backed query rewriter."""

        self.llm_client = llm_client

    def rewrite(self, query: str) -> str:
        """Rewrite a query while preserving user intent."""

        if self.llm_client is None:
            rewritten = query.strip()
            logger.debug("QueryRewriter fallback used original query")
            return rewritten

        prompt = (
            "Rewrite the user query for search relevance while preserving intent.\n"
            f"Query: {query}\n"
            "Return one rewritten query."
        )
        rewritten = self.llm_client.complete(prompt=prompt).strip()
        if rewritten.lower().startswith("refined query:"):
            rewritten = rewritten.split(":", 1)[1].strip()
        logger.info("QueryRewriter output: %s", rewritten)
        return rewritten or query
