"""Self-reflective RAG orchestration with retrieval gating."""

from __future__ import annotations

import logging
from typing import Optional

from raglib.components.decision import RETRIEVAL_NONE, DecisionEngine
from raglib.components.generator import Generator
from raglib.components.refiner import Refiner
from raglib.components.reflection import (
    ACTION_RETRY,
    ACTION_SUFFICIENT,
    ACTION_WEB_FALLBACK,
    ReflectionModule,
)
from raglib.components.retriever import Retriever
from raglib.components.web_retriever import WebRetriever
from raglib.core.base import BaseRAG
from raglib.schemas import GenerationResult

logger = logging.getLogger(__name__)


class SelfRAG(BaseRAG):
    """Runs decision-gated retrieval with reflection-based adaptation."""

    def __init__(
        self,
        decision_engine: Optional[DecisionEngine] = None,
        retriever: Optional[Retriever] = None,
        web_retriever: Optional[WebRetriever] = None,
        reflection_module: Optional[ReflectionModule] = None,
        refiner: Optional[Refiner] = None,
        generator: Optional[Generator] = None,
        **kwargs,
    ):
        """Initialize SelfRAG with decision, reflection, and retrieval components."""

        super().__init__(
            decision_engine=decision_engine,
            retriever=retriever,
            web_retriever=web_retriever,
            reflection_module=reflection_module,
            refiner=refiner,
            generator=generator,
            **kwargs,
        )

    def run(self, query: str) -> GenerationResult:
        """Run conditional retrieval, reflection, and final generation."""

        if self.generator is None:
            raise ValueError("SelfRAG requires a generator")

        active_query = self.pre_retrieve(query)
        trace = []
        documents = []

        route = RETRIEVAL_NONE
        if self.decision_engine is not None:
            should_retrieve = self.decision_engine.should_retrieve(active_query)
            route = self.decision_engine.retrieval_type(active_query)
            logger.info("SelfRAG decision should_retrieve=%s route=%s", should_retrieve, route)
            trace.append(f"decision:{route}")
            if should_retrieve and route != RETRIEVAL_NONE:
                documents = self._retrieve_by_mode(active_query, route)
        elif self.retriever is not None:
            documents = self.retriever.retrieve(active_query)

        if self.reflection_module is not None:
            action = self.reflection_module.reflect(active_query, documents)
            trace.append(f"reflection:{action}")
            if action == ACTION_RETRY and self.refiner is not None and self.retriever is not None:
                active_query = self.refiner.refine(active_query, documents)
                documents = self.retriever.retrieve(active_query)
            elif action == ACTION_WEB_FALLBACK and self.web_retriever is not None:
                documents = self.web_retriever.retrieve(active_query)
            elif action == ACTION_SUFFICIENT:
                logger.info("SelfRAG reflection deemed context sufficient")

        documents = self.post_retrieve(active_query, documents)
        documents = self.pre_generate(active_query, documents)
        result = self.generator.generate(query=active_query, documents=documents, reasoning_trace=trace)
        return self.post_generate(result)
