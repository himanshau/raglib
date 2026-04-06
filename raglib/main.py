"""Executable demo for running raglib end-to-end workflows."""

from __future__ import annotations

import logging
from typing import List

from raglib.components.context_reducer import ContextReducer
from raglib.components.deduplicator import Deduplicator
from raglib.components.evaluator import Evaluator
from raglib.components.generator import Generator
from raglib.components.hybrid_retriever import HybridRetriever
from raglib.components.memory import MemoryModule
from raglib.components.planner import Planner
from raglib.components.query_expander import QueryExpander
from raglib.components.query_rewriter import QueryRewriter
from raglib.components.refiner import Refiner
from raglib.components.reranker import Reranker
from raglib.components.retriever import Retriever
from raglib.components.router_retriever import RouterRetriever
from raglib.components.web_retriever import WebRetriever
from raglib.llm.mock_client import MockLLMClient
from raglib.providers import (
    BaseSearchProvider,
    DuckDuckGoProvider,
    ProviderChain,
    ProviderError,
    TavilyProvider,
)
from raglib.rag_types.advanced_rag import AdvancedRAG
from raglib.rag_types.agentic_rag import AgenticRAG
from raglib.rag_types.corrective_rag import CorrectiveRAG
from raglib.rag_types.memory_rag import MemoryRAG
from raglib.rag_types.naive_rag import NaiveRAG
from raglib.rag_types.self_rag import SelfRAG
from raglib.schemas import Document, GenerationResult

logger = logging.getLogger(__name__)

DEFAULT_LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
SAMPLE_QUERY: str = "How can a production RAG system reduce hallucinations?"


class StaticFallbackProvider(BaseSearchProvider):
    """Provides deterministic local web-like results when APIs are unavailable."""

    PROVIDER_NAME: str = "static_fallback"

    def __init__(self, corpus: List[Document]):
        """Initialize fallback provider with a local corpus."""

        self.corpus = corpus

    @property
    def name(self) -> str:
        """Return the provider identifier."""

        return self.PROVIDER_NAME

    def search(self, query: str, num_results: int = 5) -> List[Document]:
        """Return deterministic local documents for fallback use."""

        selected = self.corpus[:num_results]
        docs: List[Document] = []
        for idx, doc in enumerate(selected):
            docs.append(
                Document(
                    id=f"fallback-{idx}-{doc.id}",
                    content=f"Fallback result for '{query}': {doc.content}",
                    metadata={**doc.metadata, "provider": self.name},
                    score=0.5,
                    source=self.name,
                )
            )
        logger.info("StaticFallbackProvider returned %d documents", len(docs))
        return docs


def configure_logging(level: str = "INFO") -> None:
    """Configure application logging for demo visibility."""

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=DEFAULT_LOG_FORMAT)


def build_corpus() -> List[Document]:
    """Build a varied local corpus for vector retrieval."""

    return [
        Document(
            id="doc-1",
            content="RAG pipelines reduce hallucinations by grounding generation on retrieved documents.",
            metadata={"topic": "rag", "type": "overview"},
            source="vector",
        ),
        Document(
            id="doc-2",
            content="Reranking improves relevance by re-scoring candidate chunks with query-aware features.",
            metadata={"topic": "reranking", "type": "technique"},
            source="vector",
        ),
        Document(
            id="doc-3",
            content="Evaluation filters low-signal context and supports corrective retrieval retries.",
            metadata={"topic": "evaluation", "type": "quality"},
            source="vector",
        ),
        Document(
            id="doc-4",
            content="Context reduction preserves the highest-value facts within strict prompt token budgets.",
            metadata={"topic": "context", "type": "optimization"},
            source="vector",
        ),
        Document(
            id="doc-5",
            content="Hybrid retrieval combines internal embeddings with web search for freshness.",
            metadata={"topic": "hybrid", "type": "architecture"},
            source="vector",
        ),
        Document(
            id="doc-6",
            content="Memory-augmented RAG carries prior turns to improve follow-up answer consistency.",
            metadata={"topic": "memory", "type": "conversation"},
            source="vector",
        ),
        Document(
            id="doc-7",
            content="Agentic RAG decomposes complex goals into ordered retrieval steps and synthesis.",
            metadata={"topic": "agentic", "type": "planning"},
            source="vector",
        ),
        Document(
            id="doc-8",
            content="Query rewriting and expansion improve recall by matching diverse phrasings.",
            metadata={"topic": "query", "type": "preprocessing"},
            source="vector",
        ),
        Document(
            id="doc-9",
            content="Tool-augmented systems merge retrieval context with calculator or database outputs.",
            metadata={"topic": "tools", "type": "integration"},
            source="vector",
        ),
    ]


def print_result(name: str, result: GenerationResult) -> None:
    """Print a concise user-facing summary of one run result."""

    source_ids = ", ".join(doc.id for doc in result.sources[:3]) if result.sources else "none"
    print(f"\n{name} Answer:\n{result.answer}")
    print(f"{name} Sources: {source_ids}")


def run_demo() -> None:
    """Run all requested demonstrations in sequence."""

    configure_logging(level="INFO")

    corpus = build_corpus()
    llm_client = MockLLMClient()

    duckduckgo_provider = DuckDuckGoProvider()
    tavily_provider = TavilyProvider(api_key=None)
    static_fallback_provider = StaticFallbackProvider(corpus=corpus)

    retriever = Retriever(documents=corpus, top_k=5)
    web_retriever = WebRetriever(provider=duckduckgo_provider, top_k=3)
    hybrid_retriever = HybridRetriever(vector_retriever=retriever, web_retriever=web_retriever)
    reranker = Reranker(top_k=3)
    evaluator = Evaluator(relevance_threshold=0.3)
    refiner = Refiner(llm_client=llm_client, max_retries=2)
    memory_module = MemoryModule(max_turns=10)
    planner = Planner(llm_client=llm_client)
    generator = Generator(llm_client=llm_client, max_context_tokens=350)
    query_expander = QueryExpander(llm_client=llm_client, max_variants=4)
    query_rewriter = QueryRewriter(llm_client=llm_client)
    deduplicator = Deduplicator(similarity_threshold=0.82)
    context_reducer = ContextReducer(max_context_tokens=300)
    router_retriever = RouterRetriever(
        vector_retriever=retriever,
        web_retriever=web_retriever,
        hybrid_retriever=hybrid_retriever,
        llm_client=llm_client,
    )

    logger.info("Instantiated all required components successfully")
    logger.info("Query rewriter sample: %s", query_rewriter.rewrite(SAMPLE_QUERY))
    logger.info("Query expander sample size: %d", len(query_expander.expand(SAMPLE_QUERY)))
    logger.info("Router route sample: %s", router_retriever.route(SAMPLE_QUERY))

    print("\nWeb provider count in the library:")
    print("Total web providers: 9")
    print("Free default: duckduckgo")
    print("Auth-required: tavily, serpapi, brave, bing, google_cse, exa, searxng")
    print("Offline/local: local")

    naive_rag = NaiveRAG(retriever=retriever, generator=generator)
    naive_result = naive_rag.run(SAMPLE_QUERY)
    print_result("NaiveRAG", naive_result)

    advanced_rag = AdvancedRAG(
        retriever=retriever,
        reranker=reranker,
        context_reducer=context_reducer,
        deduplicator=deduplicator,
        generator=generator,
    )
    advanced_result = advanced_rag.run(SAMPLE_QUERY)
    print_result("AdvancedRAG", advanced_result)

    strict_evaluator = Evaluator(relevance_threshold=1.1)
    corrective_rag = CorrectiveRAG(
        retriever=retriever,
        evaluator=strict_evaluator,
        refiner=refiner,
        generator=generator,
        max_retries=1,
    )
    corrective_result = corrective_rag.run("Explain trust calibration in retrieval systems")
    print_result("CorrectiveRAG", corrective_result)

    self_rag = SelfRAG(
        retriever=retriever,
        generator=generator,
        llm_client=llm_client,
        evaluator=evaluator,
        refiner=refiner,
        query_rewriter=query_rewriter,
    )
    self_result = self_rag.run("What is retrieval augmented generation?")
    print_result("SelfRAG", self_result)

    agentic_rag = AgenticRAG(
        planner=planner,
        retriever=retriever,
        deduplicator=deduplicator,
        generator=generator,
    )
    agentic_result = agentic_rag.run(
        "Compare naive and corrective RAG, then recommend when to use each."
    )
    print_result("AgenticRAG", agentic_result)

    memory_rag = MemoryRAG(memory_module=memory_module, retriever=retriever, generator=generator)
    memory_turn_1 = memory_rag.run("What does reranking do in a RAG pipeline?")
    print_result("MemoryRAG Turn 1", memory_turn_1)

    memory_turn_2 = memory_rag.run("How is that different from evaluation?")
    print_result("MemoryRAG Turn 2", memory_turn_2)

    swapped_web_retriever = WebRetriever(provider=tavily_provider, top_k=3)
    logger.info(
        "Provider swap demo: original=%s swapped=%s",
        web_retriever.provider.name,
        swapped_web_retriever.provider.name,
    )

    provider_chain = ProviderChain(
        providers=[tavily_provider, duckduckgo_provider, static_fallback_provider]
    )
    chain_web_retriever = WebRetriever(provider=provider_chain, top_k=2)
    try:
        chain_docs = chain_web_retriever.retrieve("latest RAG benchmark trends")
    except ProviderError as exc:
        logger.warning("Provider chain failed unexpectedly: %s", exc)
        chain_docs = static_fallback_provider.search("latest RAG benchmark trends", num_results=2)

    print("\nProvider swap demo: DuckDuckGoProvider -> TavilyProvider(api_key=None)")
    print(f"ProviderChain returned {len(chain_docs)} documents using fallback order.")


if __name__ == "__main__":
    run_demo()
