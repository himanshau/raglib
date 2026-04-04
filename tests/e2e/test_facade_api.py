"""End-to-end tests for the high-level RAG facade API."""

from __future__ import annotations

from pathlib import Path

import pytest

from raglib import RAG
from raglib.schemas import GenerationResult

pytestmark = pytest.mark.e2e


def test_facade_rag_minimal_init_and_query_with_raw_text() -> None:
    """Verify RAG facade answers a query from raw text with minimal setup."""

    rag = RAG("RAG combines retrieval and generation.")
    result = rag.query("What does this text describe?")

    assert isinstance(result, GenerationResult)
    assert result.answer.strip() != ""


def test_facade_rag_loads_folder_of_text_files(tmp_path: Path) -> None:
    """Verify RAG facade loads text files from a folder path source."""

    (tmp_path / "a.txt").write_text("Python is a programming language.", encoding="utf-8")
    (tmp_path / "b.md").write_text("RAG can improve grounded responses.", encoding="utf-8")

    rag = RAG(str(tmp_path), top_k=4)
    result = rag.query("What is mentioned about RAG?")

    assert isinstance(result, GenerationResult)
    assert len(result.sources) >= 1


def test_facade_rag_add_ingests_additional_source() -> None:
    """Verify RAG facade add method ingests more content after initialization."""

    rag = RAG("Initial content about systems.")
    before = len(rag.retriever._documents)

    rag.add("Additional content about architecture.")
    after = len(rag.retriever._documents)

    assert after > before


def test_facade_rag_supports_rag_type_override() -> None:
    """Verify RAG facade supports selecting a non-default rag_type."""

    rag = RAG("Retrieval quality can be refined.", rag_type="corrective")
    result = rag.query("How can retrieval be improved?")

    assert isinstance(result, GenerationResult)
