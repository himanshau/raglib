"""Query refinement component for corrective retrieval loops."""

from __future__ import annotations

import logging
from typing import List

from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document

logger = logging.getLogger(__name__)


class Refiner:
    """Refines user queries after poor retrieval quality."""

    def __init__(self, llm_client: BaseLLMClient, max_retries: int = 2):
        """Initialize the refiner with an LLM client and retry policy."""

        self.llm_client = llm_client
        self.max_retries = max_retries

    def refine(self, query: str, failed_docs: List[Document]) -> str:
        """Generate a refined query from failed retrieval context."""

        snippets = "\n".join(doc.content[:180] for doc in failed_docs[:3])
        prompt = (
            "Rewrite the query to improve retrieval quality.\n"
            f"Original query: {query}\n"
            f"Low-quality snippets:\n{snippets}\n"
            "Return only the refined query."
        )
        refined = self.llm_client.complete(prompt=prompt).strip()
        if refined.lower().startswith("refined query:"):
            refined = refined.split(":", 1)[1].strip()
        logger.info("Refiner transformed query from '%s' to '%s'", query, refined)
        return refined or query
