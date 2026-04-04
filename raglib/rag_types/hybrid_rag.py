"""Hybrid RAG orchestration combining vector and web retrieval."""

from __future__ import annotations

import logging
from typing import Optional

from raglib.components.generator import Generator
from raglib.components.hybrid_retriever import HybridRetriever
from raglib.components.reranker import Reranker
from raglib.core.base import BaseRAG
from raglib.schemas import GenerationResult

logger = logging.getLogger(__name__)


class HybridRAG(BaseRAG):
    """Runs hybrid retrieval, reranking, and answer generation."""

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Reranker] = None,
        generator: Optional[Generator] = None,
        **kwargs,
    ):
        """Initialize HybridRAG with hybrid retrieval pipeline components."""

        super().__init__(
            hybrid_retriever=hybrid_retriever,
            reranker=reranker,
            generator=generator,
            **kwargs,
        )

    def run(self, query: str) -> GenerationResult:
        """Run hybrid retrieval followed by reranking and generation."""

        if self.hybrid_retriever is None or self.generator is None:
            raise ValueError("HybridRAG requires hybrid_retriever and generator")

        active_query = self.pre_retrieve(query)
        documents = self.hybrid_retriever.retrieve(active_query)
        if self.reranker is not None:
            documents = self.reranker.rerank(active_query, documents)

        documents = self.post_retrieve(active_query, documents)
        documents = self.pre_generate(active_query, documents)
        result = self.generator.generate(
            query=active_query,
            documents=documents,
            reasoning_trace=["hybrid_retrieve_rerank_generate"],
        )
        return self.post_generate(result)
