"""Document reranking utilities for post-retrieval relevance sorting."""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import List, Set

from raglib.schemas import Document

logger = logging.getLogger(__name__)


class Reranker:
    """Reranks candidate documents using lightweight lexical scoring."""

    def __init__(self, top_k: int = 3):
        """Initialize reranker settings."""

        self.top_k = top_k

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """Return top-k reranked documents for a query."""

        if not documents:
            return []

        query_terms = self._tokenize(query)
        rescored: List[Document] = []
        for doc in documents:
            score = self._score(query_terms, doc)
            rescored.append(replace(doc, score=score))

        rescored.sort(key=lambda doc: doc.score, reverse=True)
        logger.info("Reranker reduced %d docs to top %d", len(documents), self.top_k)
        return rescored[: self.top_k]

    def _score(self, query_terms: Set[str], doc: Document) -> float:
        """Score a document using overlap and length normalization."""

        doc_terms = self._tokenize(doc.content)
        if not doc_terms:
            return 0.0

        overlap = len(query_terms & doc_terms)
        overlap_ratio = overlap / max(len(query_terms), 1)
        length_norm = min(len(doc_terms), 300) / 300.0
        return 0.75 * overlap_ratio + 0.25 * length_norm

    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into a set of lowercase lexical units."""

        return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
