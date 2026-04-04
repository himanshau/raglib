"""Factory for selecting vector store backends."""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from raglib.vectorstores.base_store import BaseVectorStore
from raglib.vectorstores.chroma_store import ChromaVectorStore
from raglib.vectorstores.in_memory_store import InMemoryVectorStore

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """Build vector store instances from backend name or prebuilt instance."""

    @staticmethod
    def _is_chroma_panic(exc: BaseException) -> bool:
        """Return True when an exception is a Chroma Rust panic wrapper."""

        return exc.__class__.__name__ == "PanicException"

    @classmethod
    def _build_chroma_with_recovery(
        cls,
        options: Dict[str, Any],
        fallback_to_memory: bool,
        recover_with_fresh_persist: bool,
    ) -> BaseVectorStore:
        """Build Chroma backend with panic recovery and safe fallback behavior."""

        try:
            store = ChromaVectorStore(**options)
            logger.info("VectorStoreFactory selected backend='chroma'")
            return store
        except ImportError as exc:
            if fallback_to_memory:
                logger.warning(
                    "ChromaDB is not installed; falling back to InMemoryVectorStore. Error: %s",
                    exc,
                )
                return InMemoryVectorStore()
            raise
        except Exception as exc:
            if fallback_to_memory:
                logger.warning(
                    "Chroma initialization failed; falling back to InMemoryVectorStore. Error: %s",
                    exc,
                )
                return InMemoryVectorStore()
            raise
        except BaseException as exc:
            if not cls._is_chroma_panic(exc):
                raise

            logger.warning("Detected Chroma runtime panic during initialization: %s", exc)

            persist_directory = options.get("persist_directory")
            if recover_with_fresh_persist and isinstance(persist_directory, str) and persist_directory.strip():
                base_path = Path(persist_directory)
                recovered_path = base_path.parent / f"{base_path.name}_recovered_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                retry_options = dict(options)
                retry_options["persist_directory"] = str(recovered_path)

                try:
                    logger.warning(
                        "Retrying Chroma with fresh persist directory: %s",
                        retry_options["persist_directory"],
                    )
                    store = ChromaVectorStore(**retry_options)
                    logger.info("VectorStoreFactory selected backend='chroma' (recovered persist dir)")
                    return store
                except BaseException as retry_exc:
                    if not cls._is_chroma_panic(retry_exc):
                        if fallback_to_memory:
                            logger.warning(
                                "Chroma recovery failed; falling back to InMemoryVectorStore. Error: %s",
                                retry_exc,
                            )
                            return InMemoryVectorStore()
                        raise

                    if fallback_to_memory:
                        logger.warning(
                            "Chroma recovery still panicked; falling back to InMemoryVectorStore. Error: %s",
                            retry_exc,
                        )
                        return InMemoryVectorStore()
                    raise

            if fallback_to_memory:
                logger.warning("Chroma panic detected; falling back to InMemoryVectorStore.")
                return InMemoryVectorStore()

            raise

    @staticmethod
    def build(
        vector_db: Optional[Union[str, BaseVectorStore]] = None,
        vector_db_kwargs: Optional[Dict[str, Any]] = None,
    ) -> BaseVectorStore:
        """Return configured vector store backend with Chroma default selection."""

        if isinstance(vector_db, BaseVectorStore):
            logger.info("Using prebuilt vector store instance: %s", type(vector_db).__name__)
            return vector_db

        backend = (vector_db or "chroma").strip().lower() if isinstance(vector_db, str) or vector_db is None else "chroma"
        options = dict(vector_db_kwargs or {})
        fallback_to_memory = bool(options.pop("fallback_to_memory", True))
        recover_with_fresh_persist = bool(options.pop("recover_with_fresh_persist", True))

        if backend in {"chroma", "chromadb"}:
            return VectorStoreFactory._build_chroma_with_recovery(
                options=options,
                fallback_to_memory=fallback_to_memory,
                recover_with_fresh_persist=recover_with_fresh_persist,
            )

        if backend in {"memory", "in_memory", "inmemory", "mock"}:
            logger.info("VectorStoreFactory selected backend='memory'")
            return InMemoryVectorStore()

        raise ValueError(
            f"Unknown vector_db '{backend}'. Supported values: ['chroma', 'memory'] "
            "or pass a BaseVectorStore instance."
        )
