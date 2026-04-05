"""Faithful Corrective RAG orchestration with three-path routing."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from raglib.components.evaluator import Evaluator
from raglib.components.generator import Generator
from raglib.components.refiner import Refiner
from raglib.components.retriever import Retriever
from raglib.components.web_retriever import WebRetriever
from raglib.core.base import BaseRAG
from raglib.llm.base_client import BaseLLMClient
from raglib.schemas import Document, GenerationResult

logger = logging.getLogger(__name__)

# CRAG retrieval evaluator verdicts
VERDICT_CORRECT = "correct"
VERDICT_AMBIGUOUS = "ambiguous"
VERDICT_INCORRECT = "incorrect"


class CorrectiveRAG(BaseRAG):
    """Three-path CRAG pipeline with LLM evaluation and selective context building."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        evaluator: Optional[Evaluator] = None,
        refiner: Optional[Refiner] = None,
        generator: Optional[Generator] = None,
        web_retriever: Optional[WebRetriever] = None,
        llm_client: Optional[BaseLLMClient] = None,
        max_retries: int = 2,
        **kwargs,
    ):
        """Initialize CRAG dependencies used across the three routing paths."""

        super().__init__(
            retriever=retriever,
            evaluator=evaluator,
            refiner=refiner,
            generator=generator,
            web_retriever=web_retriever,
            llm_client=llm_client,
            max_retries=max_retries,
            **kwargs,
        )

    def run(self, query: str) -> GenerationResult:
        """Execute CRAG: retrieve, evaluate, route, refine/search, then generate."""

        if self.generator is None or self.retriever is None:
            raise ValueError("CorrectiveRAG requires at least retriever and generator.")

        trace: List[str] = ["corrective_started"]
        active_query = self.pre_retrieve(query)
        documents = self.retriever.retrieve(active_query)
        documents = self.post_retrieve(active_query, documents)
        logger.info("CRAG retrieved %d documents.", len(documents))

        verdict, scored_docs = self._evaluate_with_llm(active_query, documents)
        trace.append(f"evaluation:{verdict}")
        logger.info("CRAG verdict=%s for query='%s'", verdict, active_query)

        k_in: List[Document] = []
        k_ex: List[Document] = []

        if verdict == VERDICT_CORRECT:
            k_in = self._knowledge_refinement(active_query, scored_docs)
            trace.append("path:knowledge_refinement_only")
        elif verdict == VERDICT_AMBIGUOUS:
            k_in = self._knowledge_refinement(active_query, scored_docs)
            k_ex = self._knowledge_searching(active_query, trace)
            trace.append("path:refinement_plus_web")
        else:
            k_ex = self._knowledge_searching(active_query, trace)
            trace.append("path:web_only")

        final_docs = self._merge_context(k_in, k_ex, verdict)
        logger.info("CRAG generating with k_in=%d k_ex=%d docs.", len(k_in), len(k_ex))

        final_docs = self.pre_generate(active_query, final_docs)
        result = self.generator.generate(
            query=query,
            documents=final_docs,
            reasoning_trace=trace,
        )
        return self.post_generate(result)

    def _evaluate_with_llm(self, query: str, documents: List[Document]) -> Tuple[str, List[Document]]:
        """Evaluate retrieval quality using LLM labels with keyword fallback."""

        if not documents:
            logger.warning("CRAG received empty document list - verdict: incorrect.")
            return VERDICT_INCORRECT, []

        if self.llm_client is None:
            logger.warning(
                "CRAG: no llm_client provided. Falling back to keyword evaluation. "
                "Pass llm_client from the main chat model for full CRAG behavior."
            )
            return self._fallback_keyword_verdict(query, documents)

        scores: List[Tuple[Document, str]] = []
        for doc in documents:
            score = self._score_single_document(query, doc)
            scores.append((doc, score))
            logger.debug("CRAG doc=%s score=%s", doc.id, score)

        correct_docs = [doc for doc, label in scores if label == VERDICT_CORRECT]
        ambiguous_docs = [doc for doc, label in scores if label == VERDICT_AMBIGUOUS]
        incorrect_docs = [doc for doc, label in scores if label == VERDICT_INCORRECT]

        if correct_docs and not incorrect_docs:
            return VERDICT_CORRECT, correct_docs + ambiguous_docs
        if correct_docs or ambiguous_docs:
            return VERDICT_AMBIGUOUS, correct_docs + ambiguous_docs
        return VERDICT_INCORRECT, []

    def _score_single_document(self, query: str, document: Document) -> str:
        """Label one document as correct, ambiguous, or incorrect for the query."""

        llm_client = self.llm_client
        if llm_client is None:
            return VERDICT_AMBIGUOUS

        snippet = document.content[:600].strip()
        prompt = (
            "You are a retrieval quality evaluator.\n\n"
            f"Query: {query}\n\n"
            f"Retrieved document:\n{snippet}\n\n"
            "Is this document relevant and accurate for answering the query?\n"
            "Reply with exactly one word:\n"
            "  CORRECT - the document directly and accurately answers the query.\n"
            "  AMBIGUOUS - the document is partially relevant or uncertain.\n"
            "  INCORRECT - the document is irrelevant or misleading for this query.\n\n"
            "Your answer (one word only):"
        )

        try:
            response = llm_client.complete(prompt).strip().upper()
        except Exception as exc:  # noqa: BLE001
            logger.warning("CRAG LLM evaluator call failed: %s - defaulting to AMBIGUOUS.", exc)
            return VERDICT_AMBIGUOUS

        if "CORRECT" in response and "INCORRECT" not in response:
            return VERDICT_CORRECT
        if "INCORRECT" in response:
            return VERDICT_INCORRECT
        return VERDICT_AMBIGUOUS

    def _knowledge_refinement(self, query: str, documents: List[Document]) -> List[Document]:
        """Decompose, filter, and recompose retrieved context into refined k_in."""

        if not documents:
            return []

        refined: List[Document] = []
        for document in documents:
            strips = self._decompose_into_strips(document.content)

            if self.llm_client is not None:
                kept_strips = self._filter_strips_with_llm(query, strips)
            else:
                query_words = set(query.lower().split())
                kept_strips = [
                    strip for strip in strips if query_words & set(strip.lower().split())
                ] or strips

            if not kept_strips:
                continue

            refined_content = " ".join(kept_strips)
            refined.append(
                Document(
                    id=f"{document.id}_refined",
                    content=refined_content,
                    metadata={**document.metadata, "refined": True},
                    score=document.score,
                    source=document.source,
                )
            )
            logger.debug(
                "CRAG refined doc=%s: %d/%d strips kept.",
                document.id,
                len(kept_strips),
                len(strips),
            )

        logger.info("CRAG knowledge refinement: %d -> %d docs.", len(documents), len(refined))
        return refined

    def _decompose_into_strips(self, text: str) -> List[str]:
        """Split text into sentence strips without extra heavy dependencies."""

        import re

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [sentence.strip() for sentence in sentences if len(sentence.strip()) >= 10]

    def _filter_strips_with_llm(self, query: str, strips: List[str]) -> List[str]:
        """Batch filter sentence strips by relevance using the LLM."""

        if not strips:
            return []

        llm_client = self.llm_client
        if llm_client is None:
            return strips

        numbered_strips = "\n".join(f"{idx + 1}. {strip}" for idx, strip in enumerate(strips))
        prompt = (
            f"Query: {query}\n\n"
            f"Knowledge strips:\n{numbered_strips}\n\n"
            "Which strip numbers are relevant to the query? "
            "Reply with only the relevant numbers separated by commas (e.g. 1,3,5). "
            "If all are relevant reply ALL. If none reply NONE."
        )

        try:
            response = llm_client.complete(prompt).strip().upper()
        except Exception as exc:  # noqa: BLE001
            logger.warning("CRAG strip filter LLM call failed: %s - keeping all strips.", exc)
            return strips

        if response == "ALL":
            return strips
        if response == "NONE":
            return []

        import re

        numbers = re.findall(r"\d+", response)
        kept: List[str] = []
        for raw_number in numbers:
            index = int(raw_number) - 1
            if 0 <= index < len(strips):
                kept.append(strips[index])

        return kept if kept else strips

    def _knowledge_searching(self, query: str, trace: List[str]) -> List[Document]:
        """Rewrite and search web for external context k_ex with vector fallback."""

        if self.web_retriever is None:
            logger.warning(
                "CRAG: web_retriever not configured. Falling back to vector refiner retry."
            )
            return self._vector_refiner_fallback(query, trace)

        search_query = self._rewrite_for_web_search(query)
        trace.append(f"web_search_query:{search_query[:60]}")
        logger.info("CRAG web search query: '%s'", search_query)

        try:
            web_documents = self.web_retriever.retrieve(search_query)
            logger.info("CRAG web search returned %d documents.", len(web_documents))
            trace.append(f"web_results:{len(web_documents)}")
            return web_documents
        except Exception as exc:  # noqa: BLE001
            logger.error("CRAG web search failed: %s", exc)
            trace.append("web_search_failed")
            return []

    def _rewrite_for_web_search(self, query: str) -> str:
        """Rewrite the user query into an effective web-search query string."""

        if self.llm_client is None:
            return query

        prompt = (
            "Rewrite the following question as a concise, effective web search query.\n"
            "Return only the search query string - no explanation, no quotes.\n\n"
            f"Question: {query}\n\n"
            "Search query:"
        )

        try:
            rewritten = self.llm_client.complete(prompt).strip()
            if 5 <= len(rewritten) <= 200:
                return rewritten
        except Exception as exc:  # noqa: BLE001
            logger.warning("CRAG query rewrite failed: %s - using original.", exc)

        return query

    def _merge_context(self, k_in: List[Document], k_ex: List[Document], verdict: str) -> List[Document]:
        """Build generator context according to CRAG verdict routing semantics."""

        if verdict == VERDICT_CORRECT:
            merged = k_in
        elif verdict == VERDICT_AMBIGUOUS:
            merged = k_in + k_ex
        else:
            merged = k_ex

        seen_ids = set()
        final_docs: List[Document] = []
        for document in merged:
            if document.id in seen_ids:
                continue
            seen_ids.add(document.id)
            final_docs.append(document)
        return final_docs

    def _fallback_keyword_verdict(self, query: str, documents: List[Document]) -> Tuple[str, List[Document]]:
        """Fallback verdict route based on keyword evaluator when LLM is unavailable."""

        if self.evaluator is not None:
            filtered = self.evaluator.evaluate(query, documents)
            if len(filtered) == len(documents):
                return VERDICT_CORRECT, filtered
            if filtered:
                return VERDICT_AMBIGUOUS, filtered
            return VERDICT_INCORRECT, []
        return VERDICT_AMBIGUOUS, documents

    def _vector_refiner_fallback(self, query: str, trace: List[str]) -> List[Document]:
        """Fallback to vector retriever retries when web retrieval is unavailable."""

        if self.refiner is None or self.retriever is None:
            logger.warning("CRAG: no web_retriever and no usable refiner/retriever fallback.")
            return []

        retries = 0
        active_query = query
        while retries < self.max_retries:
            retries += 1
            active_query = self.refiner.refine(active_query, [])
            trace.append(f"vector_retry_{retries}")
            logger.info("CRAG vector retry %d query='%s'", retries, active_query)
            docs = self.retriever.retrieve(active_query)
            if docs:
                return docs

        return []
