"""Corrective RAG orchestration with refinement retries."""

from __future__ import annotations

import logging
from typing import Optional

from raglib.components.evaluator import Evaluator
from raglib.components.generator import Generator
from raglib.components.refiner import Refiner
from raglib.components.retriever import Retriever
from raglib.core.base import BaseRAG
from raglib.schemas import GenerationResult

logger = logging.getLogger(__name__)


class CorrectiveRAG(BaseRAG):
    """Runs corrective retrieval by refining query when quality is poor."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        evaluator: Optional[Evaluator] = None,
        refiner: Optional[Refiner] = None,
        generator: Optional[Generator] = None,
        max_retries: int = 2,
        **kwargs,
    ):
        """Initialize CorrectiveRAG with retrieval quality controls."""

        super().__init__(
            retriever=retriever,
            evaluator=evaluator,
            refiner=refiner,
            generator=generator,
            max_retries=max_retries,
            **kwargs,
        )

    def run(self, query: str) -> GenerationResult:
        """Run retrieve, evaluate, refine, retry loop, then generate."""

        if self.generator is None or self.retriever is None or self.evaluator is None:
            raise ValueError("CorrectiveRAG requires retriever, evaluator, and generator")

        active_query = self.pre_retrieve(query)
        documents = self.retriever.retrieve(active_query)
        documents = self.post_retrieve(active_query, documents)
        trace = ["corrective_started"]

        retries = 0
        while True:
            filtered = self.evaluator.evaluate(active_query, documents)
            if filtered:
                documents = filtered
                trace.append("evaluation_passed")
                break

            trace.append("evaluation_failed")
            if self.refiner is None or retries >= self.max_retries:
                break

            retries += 1
            active_query = self.refiner.refine(active_query, documents)
            logger.info("CorrectiveRAG retry=%d query=%s", retries, active_query)
            trace.append(f"refined_retry_{retries}")
            documents = self.retriever.retrieve(active_query)

        documents = self.pre_generate(active_query, documents)
        result = self.generator.generate(query=active_query, documents=documents, reasoning_trace=trace)
        return self.post_generate(result)
