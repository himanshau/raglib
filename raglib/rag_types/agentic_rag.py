"""Agentic RAG orchestration using query planning and iterative retrieval."""

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


class AgenticRAG(BaseRAG):
    """Runs a planned multi-step retrieval workflow before generation."""

    def __init__(
        self,
        planner: Optional[Planner] = None,
        retriever: Optional[Retriever] = None,
        deduplicator: Optional[Deduplicator] = None,
        generator: Optional[Generator] = None,
        **kwargs,
    ):
        """Initialize AgenticRAG with planner, retriever, and generator."""

        super().__init__(
            planner=planner,
            retriever=retriever,
            deduplicator=deduplicator,
            generator=generator,
            **kwargs,
        )

    def run(self, query: str) -> GenerationResult:
        """Plan sub-queries, retrieve per step, merge, and generate."""

        if self.planner is None or self.retriever is None or self.generator is None:
            raise ValueError("AgenticRAG requires planner, retriever, and generator")

        active_query = self.pre_retrieve(query)
        plan = self.planner.plan(active_query)
        logger.info("AgenticRAG plan=%s", plan)
        trace = [f"plan_steps:{len(plan)}"]

        merged: Dict[str, Document] = {}
        for index, sub_query in enumerate(plan, start=1):
            logger.info("AgenticRAG executing step=%d query=%s", index, sub_query)
            docs = self.retriever.retrieve(sub_query)
            for doc in docs:
                key = doc.id or f"doc-{abs(hash(doc.content.lower().strip()))}"
                if key not in merged or doc.score > merged[key].score:
                    merged[key] = doc

        documents = list(merged.values())
        if self.deduplicator is not None:
            documents = self.deduplicator.deduplicate(documents)

        documents = sorted(documents, key=lambda doc: doc.score, reverse=True)
        documents = self.post_retrieve(active_query, documents)
        documents = self.pre_generate(active_query, documents)
        result = self.generator.generate(query=active_query, documents=documents, reasoning_trace=trace)
        return self.post_generate(result)
