"""Configuration dataclasses for raglib components."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    """Configuration options for vector retrieval."""

    top_k: int = 5
    similarity_threshold: float = 0.0
    use_faiss: bool = False


@dataclass
class WebRetrieverConfig:
    """Configuration options for web retrieval."""

    provider_name: str = "duckduckgo"
    top_k: int = 5
    timeout: int = 10


@dataclass
class RAGConfig:
    """Top-level pipeline configuration for RAG orchestration."""

    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    web_retriever: WebRetrieverConfig = field(default_factory=WebRetrieverConfig)
    max_retries: int = 2
    relevance_threshold: float = 0.3
    max_context_tokens: int = 3000
    enable_logging: bool = True
    log_level: str = "INFO"
