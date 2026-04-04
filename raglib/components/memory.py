"""Conversation memory store for RAG turn history."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, List

from raglib.schemas import Document

logger = logging.getLogger(__name__)


@dataclass
class MemoryTurn:
    """Represents one conversational turn in memory."""

    query: str
    answer: str
    documents: List[Document]


class MemoryModule:
    """Stores and exposes bounded conversational memory context."""

    def __init__(self, max_turns: int = 10):
        """Initialize bounded memory storage."""

        self.max_turns = max_turns
        self._turns: Deque[MemoryTurn] = deque(maxlen=max_turns)

    def add(self, query: str, answer: str, documents: List[Document]) -> None:
        """Add a turn to memory."""

        self._turns.append(MemoryTurn(query=query, answer=answer, documents=documents))
        logger.info("MemoryModule stored turn count=%d", len(self._turns))

    def get_context(self) -> str:
        """Return serialized memory context for prompt injection."""

        if not self._turns:
            return ""

        lines: List[str] = []
        for idx, turn in enumerate(self._turns, start=1):
            lines.append(f"Turn {idx} Query: {turn.query}")
            lines.append(f"Turn {idx} Answer: {turn.answer}")
            if turn.documents:
                sources = ", ".join(doc.id for doc in turn.documents[:3])
                lines.append(f"Turn {idx} Sources: {sources}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all memory turns."""

        self._turns.clear()
        logger.info("MemoryModule cleared")
