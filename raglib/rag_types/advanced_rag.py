"""Advanced RAG orchestration with ranking and context pruning."""

from __future__ import annotations

import logging
from typing import Optional

from raglib.components.context_reducer import ContextReducer
from raglib.components.deduplicator import Deduplicator
from raglib.components.generator import Generator
from raglib.components.reranker import Reranker
from raglib.components.retriever import Retriever
from raglib.core.base import BaseRAG
from raglib.schemas import GenerationResult

logger = logging.getLogger(__name__)


class AdvancedRAG(BaseRAG):
    """Runs retrieve, rerank, reduce, deduplicate, and generate."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        reranker: Optional[Reranker] = None,
        context_reducer: Optional[ContextReducer] = None,
        deduplicator: Optional[Deduplicator] = None,
        generator: Optional[Generator] = None,
        **kwargs,
    ):
        """Initialize AdvancedRAG with injected orchestration components."""

        super().__init__(
            retriever=retriever,
            reranker=reranker,
            context_reducer=context_reducer,
            deduplicator=deduplicator,
            generator=generator,
            **kwargs,
        )

    def run(self, query: str) -> GenerationResult:
        """Run retrieve, rerank, context reduction, deduplication, and generation."""

        if self.generator is None or self.retriever is None:
            raise ValueError("AdvancedRAG requires retriever and generator")

        active_query = self.pre_retrieve(query)
        documents = self.retriever.retrieve(active_query)
        documents = self.post_retrieve(active_query, documents)

        if self.reranker is not None:
            documents = self.reranker.rerank(active_query, documents)
        if self.context_reducer is not None:
            documents = self.context_reducer.reduce(documents)
        if self.deduplicator is not None:
            documents = self.deduplicator.deduplicate(documents)

        documents = self.pre_generate(active_query, documents)
        result = self.generator.generate(
            query=active_query,
            documents=documents,
            reasoning_trace=["advanced_pipeline"],
        )
        return self.post_generate(result)
