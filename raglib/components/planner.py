"""Query decomposition planner for multi-step retrieval."""

from __future__ import annotations

import logging
import re
from typing import List

from raglib.llm.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class Planner:
    """Breaks complex user queries into simpler sub-queries."""

    def __init__(self, llm_client: BaseLLMClient):
        """Initialize planner with an LLM client."""

        self.llm_client = llm_client

    def plan(self, query: str) -> List[str]:
        """Decompose a query into ordered sub-queries."""

        prompt = (
            "Decompose this user query into concise sub-queries for retrieval.\n"
            f"Query: {query}\n"
            "Return one sub-query per line."
        )
        raw = self.llm_client.complete(prompt=prompt)
        lines = [line.strip() for line in raw.splitlines() if line.strip()]

        sub_queries: List[str] = []
        for line in lines:
            cleaned = re.sub(r"^\d+[\).:-]\s*", "", line)
            if cleaned:
                sub_queries.append(cleaned)

        if not sub_queries:
            parts = [p.strip() for p in re.split(r"\band\b|\bthen\b|\b,\b", query) if p.strip()]
            sub_queries = parts if parts else [query]

        logger.info("Planner produced %d sub-queries", len(sub_queries))
        return sub_queries
