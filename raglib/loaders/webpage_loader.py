"""Loader for webpage sources addressed by URL."""

from __future__ import annotations

import logging
import re
import urllib.request
from html.parser import HTMLParser
from typing import List

from raglib.schemas import Document

logger = logging.getLogger(__name__)


class _VisibleTextParser(HTMLParser):
    """Extracts visible text content from HTML payloads."""

    def __init__(self):
        """Initialize parser state."""

        super().__init__()
        self._skip_depth = 0
        self._chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs):
        """Track tags whose content should be ignored."""

        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str):
        """Stop ignoring ignored-tag content at end tags."""

        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str):
        """Collect visible text nodes from HTML data."""

        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)

    @property
    def text(self) -> str:
        """Return extracted visible text as one normalized string."""

        return "\n".join(self._chunks)


class WebPageLoader:
    """Loads webpage text into Document objects."""

    def load(self, url: str) -> List[Document]:
        """Load webpage content from URL into a single Document."""

        logger.info("WebPageLoader fetching url=%s", url)
        with urllib.request.urlopen(url, timeout=10) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            html = response.read().decode(charset, errors="ignore")

        parser = _VisibleTextParser()
        parser.feed(html)
        text = parser.text.strip()

        if not text:
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()

        if not text:
            logger.warning("WebPageLoader found no visible text at url=%s", url)
            return []

        return [
            Document(
                id=f"url-{abs(hash(url))}",
                content=text,
                metadata={"url": url, "loader": "webpage"},
                source="web",
            )
        ]
