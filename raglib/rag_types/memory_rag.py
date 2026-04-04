"""Memory-augmented RAG orchestration for multi-turn use."""

from __future__ import annotations

import logging
from typing import Optional

from raglib.components.generator import Generator
from raglib.components.memory import MemoryModule
from raglib.components.retriever import Retriever
from raglib.core.base import BaseRAG
from raglib.schemas import GenerationResult

logger = logging.getLogger(__name__)


class MemoryRAG(BaseRAG):
    """Injects conversational memory context into generation."""

    def __init__(
        self,
        memory_module: Optional[MemoryModule] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None,
        **kwargs,
    ):
        """Initialize MemoryRAG with memory, retrieval, and generation dependencies."""

        super().__init__(
            memory_module=memory_module,
            retriever=retriever,
            generator=generator,
            **kwargs,
        )

    def run(self, query: str) -> GenerationResult:
        """Retrieve with memory context, generate, then persist the turn."""

        if self.generator is None or self.retriever is None or self.memory_module is None:
            raise ValueError("MemoryRAG requires memory_module, retriever, and generator")

        active_query = self.pre_retrieve(query)
        memory_context = self.memory_module.get_context()
        logger.info("MemoryRAG using memory context length=%d", len(memory_context))
        documents = self.retriever.retrieve(active_query)

        documents = self.post_retrieve(active_query, documents)
        documents = self.pre_generate(active_query, documents)
        result = self.generator.generate(
            query=active_query,
            documents=documents,
            memory_context=memory_context,
            reasoning_trace=["memory_augmented"],
        )
        result = self.post_generate(result)
        self.memory_module.add(query=active_query, answer=result.answer, documents=result.sources)
        return result
