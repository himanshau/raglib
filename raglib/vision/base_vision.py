"""Abstract interface for vision-capable LLM clients."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseVisionClient(ABC):
    """Send an image to a vision-capable model and return extracted text."""

    @abstractmethod
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract readable text from an image file path."""
