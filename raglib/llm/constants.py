"""Centralized default model constants for all LLM-related providers."""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

PROVIDER_DEFAULT_CHAT_MODEL: Dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5-20251001",
    "groq": "llama3-8b-8192",
    "google": "gemini-1.5-flash",
    "ollama": "llama3",
}

PROVIDER_DEFAULT_EMBEDDING_MODEL: Dict[str, str] = {
    "openai": "text-embedding-3-small",
    "google": "models/embedding-001",
    "ollama": "nomic-embed-text",
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "mock": "mock-embedding-v1",
}

PROVIDER_DEFAULT_VISION_MODEL: Dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-opus-4-6",
    "google": "gemini-1.5-pro",
}

OLLAMA_DEFAULT_BASE_URL: str = "http://localhost:11434"
