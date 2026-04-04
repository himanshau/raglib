"""Multi-hop RAG orchestration with sequential sub-query retrieval."""

from __future__ import annotations

import logging
from typing import Dict, Optional

from raglib.components.deduplicator import Deduplicator
from raglib.components.generator import Generator
from raglib.components.planner import Planner
from raglib.components.retriever import Retriever
from raglib.core.base import BaseRAG
from raglib.schemas import Document, GenerationResult

logger = logging.getLogger(__name__)


class MultiHopRAG(BaseRAG):
    """Executes sequential retrieval hops planned from the user query."""

    def __init__(
        self,
        planner: Optional[Planner] = None,
        retriever: Optional[Retriever] = None,
        deduplicator: Optional[Deduplicator] = None,
        generator: Optional[Generator] = None,
        **kwargs,
    ):
        """Initialize MultiHopRAG with planning and retrieval dependencies."""

        super().__init__(
            planner=planner,
            retriever=retriever,
            deduplicator=deduplicator,
            generator=generator,
            **kwargs,
        )

    def run(self, query: str) -> GenerationResult:
        """Plan hops, retrieve sequentially, accumulate context, and generate."""

        if self.planner is None or self.retriever is None or self.generator is None:
            raise ValueError("MultiHopRAG requires planner, retriever, and generator")

        active_query = self.pre_retrieve(query)
        hops = self.planner.plan(active_query)
        trace = [f"hops:{len(hops)}"]

        merged: Dict[str, Document] = {}
        for hop_index, hop_query in enumerate(hops, start=1):
            logger.info("MultiHopRAG hop=%d query=%s", hop_index, hop_query)
            hop_docs = self.retriever.retrieve(hop_query)
            for doc in hop_docs:
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
