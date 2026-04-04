"""Query expansion component for multi-query retrieval."""

from __future__ import annotations

import logging
from typing import List, Optional

from raglib.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class QueryExpander:
    """Generates semantic variants of a user query."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None, max_variants: int = 4):
        """Initialize expansion settings and optional LLM client."""

        self.llm_client = llm_client
        self.max_variants = max_variants

    def expand(self, query: str) -> List[str]:
        """Expand a query into several retrieval variants."""

        if self.llm_client is None:
            variants = [
                query,
                f"{query} best practices",
                f"{query} implementation",
                f"{query} examples",
            ]
        else:
            prompt = (
                "Expand the query into semantic variants for retrieval.\n"
                f"Query: {query}\n"
                f"Return up to {self.max_variants} variants, one per line."
            )
            raw = self.llm_client.complete(prompt=prompt)
            variants = [line.strip(" -") for line in raw.splitlines() if line.strip()]
            variants.insert(0, query)

        deduped: List[str] = []
        seen = set()
        for variant in variants:
            key = variant.lower().strip()
            if key and key not in seen:
                deduped.append(variant.strip())
                seen.add(key)

        selected = deduped[: self.max_variants]
        logger.info("QueryExpander produced %d variants", len(selected))
        return selected
