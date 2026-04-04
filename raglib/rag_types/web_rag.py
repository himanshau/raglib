"""Web-first RAG orchestration using web retrieval and reranking."""

from __future__ import annotations

import logging
from typing import Optional

from raglib.components.generator import Generator
from raglib.components.reranker import Reranker
from raglib.components.web_retriever import WebRetriever
from raglib.core.base import BaseRAG
from raglib.schemas import GenerationResult

logger = logging.getLogger(__name__)


class WebRAG(BaseRAG):
    """Runs web retrieval, optional reranking, and answer generation."""

    def __init__(
        self,
        web_retriever: Optional[WebRetriever] = None,
        reranker: Optional[Reranker] = None,
        generator: Optional[Generator] = None,
        **kwargs,
    ):
        """Initialize WebRAG with web retrieval pipeline components."""

        super().__init__(
            web_retriever=web_retriever,
            reranker=reranker,
            generator=generator,
            **kwargs,
        )

    def run(self, query: str) -> GenerationResult:
        """Retrieve from web, rerank, and generate the final answer."""

        if self.web_retriever is None or self.generator is None:
            raise ValueError("WebRAG requires web_retriever and generator")

        active_query = self.pre_retrieve(query)
        documents = self.web_retriever.retrieve(active_query)
        if self.reranker is not None:
            documents = self.reranker.rerank(active_query, documents)

        documents = self.post_retrieve(active_query, documents)
        documents = self.pre_generate(active_query, documents)
        result = self.generator.generate(
            query=active_query,
            documents=documents,
            reasoning_trace=["web_retrieve_rerank_generate"],
        )
        return self.post_generate(result)
