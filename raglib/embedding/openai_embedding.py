"""OpenAI embedding implementation through LangChain adapters."""

from __future__ import annotations

import logging
from typing import List, Optional

from raglib.embedding.base_embedding import BaseEmbedding
from raglib.llm.constants import PROVIDER_DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class OpenAIEmbedding(BaseEmbedding):
    """Embed text with OpenAI through LangChain OpenAIEmbeddings."""

    def __init__(self, api_key: Optional[str], model_name: Optional[str] = None):
        """Initialize OpenAIEmbeddings with optional API key and model override."""

        try:
            from langchain_openai import OpenAIEmbeddings  # type: ignore[import-not-found]
            from pydantic import SecretStr
        except ImportError as exc:
            raise ImportError("pip install langchain-openai") from exc

        resolved_model = model_name or PROVIDER_DEFAULT_EMBEDDING_MODEL["openai"]
        secret_api_key = SecretStr(api_key) if api_key else None
        self._model = OpenAIEmbeddings(model=resolved_model, api_key=secret_api_key)
        self._dimension = 1536 if resolved_model == PROVIDER_DEFAULT_EMBEDDING_MODEL["openai"] else 0
        self._model_name = resolved_model
        logger.info("Using OpenAI embedding model='%s'", resolved_model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document strings using OpenAI embeddings."""

        logger.debug("OpenAIEmbedding embed_documents called with batch size=%d", len(texts))
        vectors = self._model.embed_documents(texts)
        if vectors:
            self._dimension = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string using OpenAI embeddings."""

        logger.debug("OpenAIEmbedding embed_query called with text length=%d", len(text))
        vector = self._model.embed_query(text)
        if vector:
            self._dimension = len(vector)
        return vector

    @property
    def dimension(self) -> int:
        """Return embedding dimensionality."""

        return self._dimension
