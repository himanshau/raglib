"""Vector store package exports."""

from __future__ import annotations

import logging

from raglib.vectorstores.base_store import BaseVectorStore, VectorHit, VectorRecord
from raglib.vectorstores.chroma_store import ChromaVectorStore
from raglib.vectorstores.factory import VectorStoreFactory
from raglib.vectorstores.in_memory_store import InMemoryVectorStore

logger = logging.getLogger(__name__)

__all__ = [
    "BaseVectorStore",
    "VectorRecord",
    "VectorHit",
    "VectorStoreFactory",
    "ChromaVectorStore",
    "InMemoryVectorStore",
]
