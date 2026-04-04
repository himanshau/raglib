"""Abstract embedding interface for raglib retrievers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger(__name__)


class BaseEmbedding(ABC):
    """Abstract interface all embedding implementations must satisfy."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of document strings."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of vectors produced by this embedding model."""
