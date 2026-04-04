"""Loader package exports for ingestion and chunking."""

from __future__ import annotations

import logging

from raglib.loaders.chunk_splitter import ChunkSplitter
from raglib.loaders.document_loader import DocumentLoader, SourceInput
from raglib.loaders.pdf_loader import PDFLoader
from raglib.loaders.pptx_loader import PPTXLoader
from raglib.loaders.text_loader import TextLoader
from raglib.loaders.web_loader import WebLoader
from raglib.loaders.word_loader import WordLoader

logger = logging.getLogger(__name__)

__all__ = [
    "SourceInput",
    "DocumentLoader",
    "ChunkSplitter",
    "TextLoader",
    "WordLoader",
    "PPTXLoader",
    "PDFLoader",
    "WebLoader",
]
