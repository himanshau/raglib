"""Naive RAG orchestration: retrieve then generate."""

from __future__ import annotations

import logging
from typing import Optional

from raglib.components.generator import Generator
from raglib.components.retriever import Retriever
from raglib.core.base import BaseRAG
from raglib.schemas import GenerationResult

logger = logging.getLogger(__name__)


class NaiveRAG(BaseRAG):
    """Runs the simplest retrieval-then-generation workflow."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None,
        **kwargs,
    ):
        """Initialize NaiveRAG with injected retriever and generator."""

        super().__init__(retriever=retriever, generator=generator, **kwargs)

    def run(self, query: str) -> GenerationResult:
        """Run retrieve then generate without post-processing stages."""

        active_query = self.pre_retrieve(query)
        documents = self.retriever.retrieve(active_query) if self.retriever is not None else []
        documents = self.post_retrieve(active_query, documents)
        documents = self.pre_generate(active_query, documents)

        if self.generator is None:
            raise ValueError("NaiveRAG requires a generator")

        result = self.generator.generate(
            query=active_query,
            documents=documents,
            memory_context="",
            reasoning_trace=["naive_retrieve_generate"],
        )
        return self.post_generate(result)
