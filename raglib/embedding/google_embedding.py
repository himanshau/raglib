"""Google embedding implementation through LangChain adapters."""

from __future__ import annotations

import logging
from typing import List, Optional

from raglib.embedding.base_embedding import BaseEmbedding
from raglib.llm.constants import PROVIDER_DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class GoogleEmbedding(BaseEmbedding):
    """Embed text with Google models through LangChain Google embeddings."""

    def __init__(self, api_key: Optional[str], model_name: Optional[str] = None):
        """Initialize GoogleGenerativeAIEmbeddings with key and optional model override."""

        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("pip install langchain-google-genai") from exc

        resolved_model = model_name or PROVIDER_DEFAULT_EMBEDDING_MODEL["google"]
        self._model = GoogleGenerativeAIEmbeddings(model=resolved_model, google_api_key=api_key)
        self._dimension = 768 if resolved_model == PROVIDER_DEFAULT_EMBEDDING_MODEL["google"] else 0
        self._model_name = resolved_model
        logger.info("Using Google embedding model='%s'", resolved_model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings using Google embeddings."""

        logger.debug("GoogleEmbedding embed_documents called with batch size=%d", len(texts))
        vectors = self._model.embed_documents(texts)
        if vectors:
            self._dimension = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string using Google embeddings."""

        logger.debug("GoogleEmbedding embed_query called with text length=%d", len(text))
        vector = self._model.embed_query(text)
        if vector:
            self._dimension = len(vector)
        return vector

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality."""

        return self._dimension
