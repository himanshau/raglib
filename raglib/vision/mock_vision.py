"""Offline mock vision client for deterministic testing."""

from __future__ import annotations

import logging

from raglib.vision.base_vision import BaseVisionClient

logger = logging.getLogger(__name__)


class MockVisionClient(BaseVisionClient):
    """Return predictable OCR-like text without external dependencies."""

    def extract_text_from_image(self, image_path: str) -> str:
        """Return mock OCR output for a provided image path."""

        logger.debug("MockVisionClient extracting text from image_path=%s", image_path)
        return f"[Mock OCR output for image: {image_path}. No vision model configured.]"
