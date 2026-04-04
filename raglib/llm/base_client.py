"""Abstract interfaces for LLM clients used by raglib."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Defines the contract for language model clients."""

    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str:
        """Generate a completion for a single prompt."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response from a structured chat transcript."""
