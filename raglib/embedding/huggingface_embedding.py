"""Free local embedding implementation powered by sentence-transformers."""

from __future__ import annotations

import logging
from typing import List, Optional

from raglib.embedding.base_embedding import BaseEmbedding
from raglib.llm.constants import PROVIDER_DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class HuggingFaceEmbedding(BaseEmbedding):
    """Provide free local embeddings via LangChain HuggingFace embeddings."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize a local sentence-transformers-backed embedding model."""

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Free local embeddings require sentence-transformers. "
                "Install with: pip install raglib[free-embed]"
            ) from exc

        resolved_model = model_name or PROVIDER_DEFAULT_EMBEDDING_MODEL["huggingface"]
        self._model = HuggingFaceEmbeddings(model_name=resolved_model)
        self._dimension = 384
        self._model_name = resolved_model
        logger.info("Using HuggingFace embedding model='%s'", resolved_model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents using local HuggingFace embeddings."""

        logger.debug("HuggingFaceEmbedding embed_documents called with batch size=%d", len(texts))
        vectors = self._model.embed_documents(texts)
        if vectors:
            self._dimension = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using local HuggingFace embeddings."""

        logger.debug("HuggingFaceEmbedding embed_query called with text length=%d", len(text))
        vector = self._model.embed_query(text)
        if vector:
            self._dimension = len(vector)
        return vector

    @property
    def dimension(self) -> int:
        """Return current embedding dimensionality."""

        return self._dimension
