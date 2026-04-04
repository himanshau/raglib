"""Chunk splitting utilities for preparing long documents for retrieval."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import List

from raglib.schemas import Document

logger = logging.getLogger(__name__)


class ChunkSplitter:
    """Splits long documents into overlapping token-based chunks."""

    def __init__(
        self,
        chunk_size_tokens: int = 500,
        chunk_overlap_tokens: int = 50,
        min_chunk_tokens: int = 20,
    ):
        """Initialize chunking parameters."""

        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be greater than zero")
        if chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens must be non-negative")
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

    def split(self, documents: List[Document]) -> List[Document]:
        """Split input documents into retrieval-friendly overlapping chunks."""

        chunked: List[Document] = []
        for document in documents:
            chunked.extend(self._split_document(document))

        logger.info("ChunkSplitter produced %d chunks from %d documents", len(chunked), len(documents))
        return chunked

    def _split_document(self, document: Document) -> List[Document]:
        """Split one document into chunked Document copies."""

        tokens = document.content.split()
        if not tokens:
            return []

        if len(tokens) <= self.chunk_size_tokens:
            parent_id = document.id or f"doc-{abs(hash(document.content))}"
            return [
                replace(
                    document,
                    id=f"{parent_id}::chunk_1",
                    metadata={
                        **document.metadata,
                        "parent_id": parent_id,
                        "chunk_index": 1,
                        "chunk_total": 1,
                    },
                    source=document.source or "text",
                    score=0.0,
                )
            ]

        chunks_text: List[str] = []
        start = 0
        total_tokens = len(tokens)

        while start < total_tokens:
            end = min(start + self.chunk_size_tokens, total_tokens)
            current = tokens[start:end]
            if len(current) >= self.min_chunk_tokens or not chunks_text:
                chunks_text.append(" ".join(current))
            if end >= total_tokens:
                break
            start = end - self.chunk_overlap_tokens

        parent_id = document.id or f"doc-{abs(hash(document.content))}"
        chunk_total = len(chunks_text)
        chunked_documents: List[Document] = []

        for index, chunk_text in enumerate(chunks_text, start=1):
            chunked_documents.append(
                replace(
                    document,
                    id=f"{parent_id}::chunk_{index}",
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "parent_id": parent_id,
                        "chunk_index": index,
                        "chunk_total": chunk_total,
                    },
                    source=document.source or "text",
                    score=0.0,
                )
            )

        return chunked_documents
