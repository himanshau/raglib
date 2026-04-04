"""Loader for plain text and markdown sources."""

from __future__ import annotations

import logging
import os
from typing import List

from raglib.schemas import Document

logger = logging.getLogger(__name__)


class PlainTextLoader:
    """Loads plain text from local files into Document objects."""

    def load(self, path: str) -> List[Document]:
        """Load a plain text file path into a single Document."""

        with open(path, "r", encoding="utf-8", errors="ignore") as file_handle:
            content = file_handle.read().strip()

        if not content:
            logger.warning("PlainTextLoader found no text in file=%s", path)
            return []

        return [
            Document(
                id=f"text-{abs(hash(os.path.abspath(path)))}",
                content=content,
                metadata={"file": path, "loader": "plain_text"},
                source="text",
            )
        ]
