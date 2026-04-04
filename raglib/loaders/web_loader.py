"""Web page loader for URL-based source ingestion."""

from __future__ import annotations

import html
import logging
import re
from typing import List
from urllib.request import Request, urlopen

from raglib.schemas import Document

logger = logging.getLogger(__name__)


class WebLoader:
    """Fetch URL content and convert it to plain text documents."""

    def load(self, url: str) -> List[Document]:
        """Load URL content and return a single extracted text Document."""

        request = Request(url, headers={"User-Agent": "raglib/0.1"})
        with urlopen(request, timeout=15) as response:
            raw_html = response.read().decode("utf-8", errors="ignore")

        text = self._html_to_text(raw_html)
        if not text:
            logger.warning("WebLoader found no text for url=%s", url)
            return []

        document = Document(
            id=f"web-{abs(hash(url))}",
            content=text,
            metadata={"url": url, "loader": "web"},
            source="web",
        )
        logger.info("WebLoader loaded 1 document from url=%s", url)
        return [document]

    def _html_to_text(self, raw_html: str) -> str:
        """Convert HTML to approximate visible plain text."""

        without_scripts = re.sub(r"<script[\\s\\S]*?</script>", " ", raw_html, flags=re.IGNORECASE)
        without_styles = re.sub(r"<style[\\s\\S]*?</style>", " ", without_scripts, flags=re.IGNORECASE)
        without_tags = re.sub(r"<[^>]+>", " ", without_styles)
        decoded = html.unescape(without_tags)
        normalized = re.sub(r"\\s+", " ", decoded).strip()
        return normalized
