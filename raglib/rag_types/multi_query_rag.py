"""Multi-query RAG orchestration with query expansion."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from raglib.components.deduplicator import Deduplicator
from raglib.components.generator import Generator
from raglib.components.query_expander import QueryExpander
from raglib.components.retriever import Retriever
from raglib.core.base import BaseRAG
from raglib.schemas import Document, GenerationResult

logger = logging.getLogger(__name__)


class MultiQueryRAG(BaseRAG):
    """Runs expanded-query retrieval and merged-context generation."""

    def __init__(
        self,
        query_expander: Optional[QueryExpander] = None,
        retriever: Optional[Retriever] = None,
        deduplicator: Optional[Deduplicator] = None,
        generator: Optional[Generator] = None,
        **kwargs,
    ):
        """Initialize MultiQueryRAG with expansion and retrieval components."""

        super().__init__(
            query_expander=query_expander,
            retriever=retriever,
            deduplicator=deduplicator,
            generator=generator,
            **kwargs,
        )

    def run(self, query: str) -> GenerationResult:
        """Expand query, retrieve per variant, merge, and generate."""

        if self.query_expander is None or self.retriever is None or self.generator is None:
            raise ValueError("MultiQueryRAG requires query_expander, retriever, and generator")

        active_query = self.pre_retrieve(query)
        variants = self.query_expander.expand(active_query)
        trace = [f"variants:{len(variants)}"]

        merged: Dict[str, Document] = {}
        for variant in variants:
            for doc in self.retriever.retrieve(variant):
                key = doc.id or f"doc-{abs(hash(doc.content.lower().strip()))}"
                if key not in merged or doc.score > merged[key].score:
                    merged[key] = doc

        documents = sorted(merged.values(), key=lambda d: d.score, reverse=True)
        if self.deduplicator is not None:
            documents = self.deduplicator.deduplicate(documents)

        documents = self.post_retrieve(active_query, documents)
        documents = self.pre_generate(active_query, documents)
        result = self.generator.generate(query=active_query, documents=documents, reasoning_trace=trace)
        return self.post_generate(result)
