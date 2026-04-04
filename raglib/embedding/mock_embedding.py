"""Deterministic mock embeddings for offline retrieval testing."""

from __future__ import annotations

import logging
import math
from typing import List

from raglib.embedding.base_embedding import BaseEmbedding

logger = logging.getLogger(__name__)


class MockEmbedding(BaseEmbedding):
    """Generate deterministic vectors without external dependencies or API keys."""

    DIMENSION = 64

    def __init__(self) -> None:
        """Initialize a deterministic embedding implementation for offline mode."""

        logger.info("Using MockEmbedding (offline mode).")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings deterministically."""

        logger.debug("MockEmbedding embed_documents called with batch size=%d", len(texts))
        return [self._hash_to_vector(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string deterministically."""

        logger.debug("MockEmbedding embed_query called with text length=%d", len(text))
        return self._hash_to_vector(text)

    @property
    def dimension(self) -> int:
        """Return the fixed embedding dimension."""

        return self.DIMENSION

    def _hash_to_vector(self, text: str) -> List[float]:
        """Map text to a normalized vector using character-frequency hashing."""

        vector = [0.0] * self.DIMENSION
        for index, character in enumerate(text.lower()):
            bucket = (ord(character) + index) % self.DIMENSION
            vector[bucket] += 1.0

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
