"""Sentence-aware chunk splitting utilities for loaded documents."""

from __future__ import annotations

import logging
import re
from typing import List

from raglib.schemas import Document

logger = logging.getLogger(__name__)


class ChunkSplitter:
    """Split documents into overlapping chunks while preserving metadata lineage."""

    def __init__(self, chunk_size: int = 400, overlap: int = 50, min_tokens: int = 20):
        """Configure chunk size, overlap, and minimal chunk token count."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if min_tokens <= 0:
            raise ValueError("min_tokens must be greater than zero")

        self._chunk_size = chunk_size
        self._overlap = overlap
        self._min_tokens = min_tokens

    def split(self, documents: List[Document]) -> List[Document]:
        """Split input documents into overlapping chunks with preserved metadata."""

        chunks: List[Document] = []
        for document in documents:
            split_texts = self._split_document(document)
            for index, chunk_text in enumerate(split_texts, start=1):
                chunk_tokens = self._count_tokens(chunk_text)
                chunks.append(
                    Document(
                        id=f"{document.id}_chunk_{index}",
                        content=chunk_text,
                        metadata={
                            **document.metadata,
                            "parent_id": document.id,
                            "chunk_index": index,
                            "chunk_tokens": chunk_tokens,
                        },
                        score=document.score,
                        source=document.source,
                    )
                )

        logger.info("ChunkSplitter created %d chunks from %d documents", len(chunks), len(documents))
        return chunks

    def _split_document(self, document: Document) -> List[str]:
        """Split one document into sentence-aware chunks with overlap."""

        text = document.content.strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            sentences = [text]

        sentence_token_counts = [self._count_tokens(sentence) for sentence in sentences]
        chunks: List[str] = []

        cursor = 0
        while cursor < len(sentences):
            start = cursor
            end = cursor
            total_tokens = 0

            while end < len(sentences):
                sentence_tokens = sentence_token_counts[end]
                if total_tokens > 0 and (total_tokens + sentence_tokens) > self._chunk_size:
                    break
                total_tokens += sentence_tokens
                end += 1
                if total_tokens >= self._chunk_size:
                    break

            if end == start:
                end = min(start + 1, len(sentences))

            chunk_text = " ".join(sentences[start:end]).strip()
            if chunk_text:
                chunks.append(chunk_text)

            if end >= len(sentences):
                break

            overlap_tokens = 0
            overlap_sentences = 0
            back = end - 1
            while back >= start and overlap_tokens < self._overlap:
                overlap_tokens += sentence_token_counts[back]
                overlap_sentences += 1
                back -= 1

            cursor = max(start + 1, end - overlap_sentences)

        if len(chunks) >= 2 and self._count_tokens(chunks[-1]) < self._min_tokens:
            chunks[-2] = f"{chunks[-2]} {chunks[-1]}".strip()
            chunks.pop()

        normalized_chunks: List[str] = []
        for chunk in chunks:
            token_count = self._count_tokens(chunk)
            if normalized_chunks and token_count < self._min_tokens:
                normalized_chunks[-1] = f"{normalized_chunks[-1]} {chunk}".strip()
                continue
            normalized_chunks.append(chunk)

        return normalized_chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentence-like units while preserving punctuation."""

        pieces = re.split(r"(?<=[.!?])\s+", text)
        return [piece.strip() for piece in pieces if piece and piece.strip()]

    def _count_tokens(self, text: str) -> int:
        """Estimate token count using whitespace-separated words."""

        return len(text.split())
