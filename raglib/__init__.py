"""Public package exports for raglib."""

from __future__ import annotations

from raglib.facade import RAG
from raglib.llm.mock_client import MockLLMClient
from raglib.schemas import Document, GenerationResult, QueryResult

__all__ = [
    "RAG",
    "Document",
    "QueryResult",
    "GenerationResult",
    "MockLLMClient",
]

__version__ = "0.1.7"
