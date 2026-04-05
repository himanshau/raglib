# raglib Usage Guide

This guide explains how to use raglib in Python, from quickstart to advanced orchestration.

## 1. What You Get

raglib gives you:

- A common document schema for retrieval and generation
- Multiple retrievers (vector, web, hybrid, multi-query, multi-hop, routing)
- Multiple RAG orchestrators (naive, advanced, corrective, self, agentic, and more)
- A mock LLM client for fully offline development and testing
- Provider abstraction and provider fallback chains

If you want to start quickly with no API keys and no external services, use `MockLLMClient` plus local `Document` objects.

## 2. Installation

Install core package:

```bash
pip install raglib-py
```

Install optional extras when needed:

```bash
pip install raglib-py[tavily]
pip install raglib-py[serpapi]
pip install raglib-py[faiss]
```

Install file format extras for automatic source loading:

```bash
pip install raglib-py[pdf,docx,pptx]
```

Install OCR extras for scanned PDFs:

```bash
pip install raglib-py[ocr]
```

## Simple entry-point API (recommended)

The simplest way to use raglib is the facade class `RAG`.

```python
from raglib import RAG

# Minimal setup: provide a source and ask questions
rag = RAG("my_documents/")
result = rag.query("What is this report about?")

print(result.answer)
print(result.sources)
```

You can also select a strategy and retrieval size while still using a simple API:

```python
from raglib import RAG

rag = RAG(
    source="my_documents/",
    rag_type="corrective",   # defaults to "naive"
    llm_key=None,             # falls back to MockLLMClient
    top_k=5,
)

result = rag.query("Summarize the key findings")
print(result.answer)
```

`source` accepts:

- A folder path
- A single file path (`.txt`, `.md`, `.docx`, `.pptx`, `.pdf`)
- A list of paths/URLs/raw strings
- A URL
- Raw text

Incremental ingestion is supported through `add()`:

```python
from raglib import RAG

rag = RAG("initial_report.pdf")
rag.add("new_data.docx")
rag.add("https://example.com/article")

result = rag.query("What changed in the new data?")
print(result.answer)
```

Advanced users can still override internals through `rag_type_kwargs`.

## 3. Core Data Types

raglib uses shared schemas across components:

- `Document`: retrieved chunk with `id`, `content`, `metadata`, `score`, `source`
- `QueryResult`: retrieval result payload with optional rewritten query and hop count
- `GenerationResult`: final answer with sources and reasoning trace

Example:

```python
from raglib.schemas import Document

doc = Document(
    id="doc_1",
    content="Python is a general-purpose programming language.",
    metadata={"topic": "programming"},
    source="vector",
)
```

## 4. Quickstart (Offline, No API Keys)

This is the fastest way to run raglib end to end.

```python
from raglib.schemas import Document
from raglib.components.retriever import Retriever
from raglib.components.generator import Generator
from raglib.llm.mock_client import MockLLMClient
from raglib.rag_types.naive_rag import NaiveRAG

# 1) Local corpus
corpus = [
    Document(id="d1", content="RAG improves factuality by grounding responses in retrieved context."),
    Document(id="d2", content="Reranking improves relevance after initial retrieval."),
    Document(id="d3", content="Evaluation can filter weak context before generation."),
]

# 2) Components
retriever = Retriever(documents=corpus, top_k=3)
llm = MockLLMClient()
generator = Generator(llm_client=llm, max_context_tokens=300)

# 3) Orchestrator
rag = NaiveRAG(retriever=retriever, generator=generator)

# 4) Run
result = rag.run("How can RAG reduce hallucinations?")
print(result.answer)
print([doc.id for doc in result.sources])
print(result.reasoning_trace)
```

## 5. Component Building Blocks

You can compose your own pipeline from these core components.

### Retrieval

- `Retriever`: in-memory vector retrieval with optional FAISS usage when installed
- `WebRetriever`: web provider adapter
- `HybridRetriever`: combines vector and web with weighted scoring
- `MultiQueryRetriever`: query expansion + merge
- `MultiHopRetriever`: planner-driven iterative retrieval
- `RouterRetriever`: route query to vector, web, or hybrid

### Quality and control

- `Reranker`: re-orders retrieved docs
- `Evaluator`: filters docs by relevance threshold
- `Refiner`: rewrites query for retries
- `DecisionEngine`: whether retrieval is needed, and what type
- `ReflectionModule`: post-retrieval decision (`sufficient`, `retry`, `web_fallback`, `give_up`)

### Context and memory

- `Deduplicator`: removes near-duplicate chunks
- `ContextReducer`: trims context to budget
- `MemoryModule`: stores previous turns and exposes memory context
- `Planner`: decomposes complex query into sub-queries

### Generation

- `Generator`: builds prompt, applies context budget, calls LLM client, returns `GenerationResult`

## 6. Standard Full Pipeline Setup

```python
from raglib.schemas import Document
from raglib.llm.mock_client import MockLLMClient

from raglib.components.retriever import Retriever
from raglib.components.reranker import Reranker
from raglib.components.evaluator import Evaluator
from raglib.components.refiner import Refiner
from raglib.components.decision import DecisionEngine
from raglib.components.reflection import ReflectionModule
from raglib.components.memory import MemoryModule
from raglib.components.planner import Planner
from raglib.components.query_expander import QueryExpander
from raglib.components.query_rewriter import QueryRewriter
from raglib.components.deduplicator import Deduplicator
from raglib.components.context_reducer import ContextReducer
from raglib.components.generator import Generator

corpus = [
    Document(id="d1", content="RAG retrieves context before generation."),
    Document(id="d2", content="Corrective RAG can retry with refined queries."),
    Document(id="d3", content="Hybrid retrieval mixes local and web knowledge."),
]

llm = MockLLMClient()

retriever = Retriever(documents=corpus, top_k=5)
reranker = Reranker(top_k=3)
evaluator = Evaluator(relevance_threshold=0.3)
refiner = Refiner(llm_client=llm, max_retries=2)
decision_engine = DecisionEngine(llm_client=llm)
reflection = ReflectionModule(evaluator=evaluator)
memory = MemoryModule(max_turns=10)
planner = Planner(llm_client=llm)
query_expander = QueryExpander(llm_client=llm, max_variants=4)
query_rewriter = QueryRewriter(llm_client=llm)
deduplicator = Deduplicator(similarity_threshold=0.85)
context_reducer = ContextReducer(max_context_tokens=300)
generator = Generator(llm_client=llm, max_context_tokens=300)
```

## 7. RAG Types and When to Use Them

- `NaiveRAG`: simplest retrieve then generate flow
- `AdvancedRAG`: retrieve -> rerank -> context reduce -> deduplicate -> generate
- `CorrectiveRAG`: retrieve -> evaluate -> refine/retry -> generate
- `SelfRAG`: decision-gated retrieval + reflection feedback
- `AgenticRAG`: planner-driven sub-query retrieval + synthesis
- `HybridRAG`: hybrid retriever + reranker + generation
- `MultiQueryRAG`: expanded query variants + merge + generation
- `MultiHopRAG`: sequential hop retrieval + accumulated context
- `RoutingRAG`: route query to best retriever path
- `MemoryRAG`: injects conversation memory into prompts
- `WebRAG`: web retrieval first, then rerank and generate
- `ToolAugmentedRAG`: retrieval + external tool hook + merged generation

## 8. Examples by Orchestrator

### 8.1 AdvancedRAG

```python
from raglib.rag_types.advanced_rag import AdvancedRAG

advanced = AdvancedRAG(
    retriever=retriever,
    reranker=reranker,
    context_reducer=context_reducer,
    deduplicator=deduplicator,
    generator=generator,
)

result = advanced.run("How do I improve retrieval relevance?")
print(result.answer)
```

### 8.2 CorrectiveRAG with retries

```python
from raglib.rag_types.corrective_rag import CorrectiveRAG

corrective = CorrectiveRAG(
    retriever=retriever,
    evaluator=evaluator,
    refiner=refiner,
    generator=generator,
    max_retries=2,
)

result = corrective.run("Explain trust calibration in retrieval")
print(result.answer)
print(result.reasoning_trace)
```

### 8.3 MemoryRAG (multi-turn)

```python
from raglib.rag_types.memory_rag import MemoryRAG

memory_rag = MemoryRAG(memory_module=memory, retriever=retriever, generator=generator)

turn1 = memory_rag.run("What does reranking do?")
turn2 = memory_rag.run("How is that different from evaluation?")

print(turn1.answer)
print(turn2.answer)
```

### 8.4 ToolAugmentedRAG

```python
from raglib.schemas import Document
from raglib.rag_types.tool_augmented_rag import ToolAugmentedRAG

def tool_call_hook(query, retrieved_docs):
    # Example: add deterministic tool output as another source
    return [
        Document(
            id="tool_1",
            content=f"Tool output for query: {query}",
            metadata={"tool": "example"},
            source="tool",
            score=0.8,
        )
    ]

rag = ToolAugmentedRAG(
    retriever=retriever,
    generator=generator,
    deduplicator=deduplicator,
    tool_call_hook=tool_call_hook,
)

result = rag.run("Summarize with external metrics")
print(result.answer)
```

## 9. Web Retrieval and Provider Chain

You can use any provider implementing `BaseSearchProvider`.

```python
from raglib.components.web_retriever import WebRetriever
from raglib.components.hybrid_retriever import HybridRetriever
from raglib.providers import DuckDuckGoProvider, TavilyProvider, ProviderChain

duckduckgo = DuckDuckGoProvider()
tavily = TavilyProvider(api_key=None)  # will raise if used directly without key

provider_chain = ProviderChain([tavily, duckduckgo])
web_retriever = WebRetriever(provider=provider_chain, top_k=5)

hybrid_retriever = HybridRetriever(
    vector_retriever=retriever,
    web_retriever=web_retriever,
    vector_weight=0.6,
    web_weight=0.4,
)

docs = hybrid_retriever.retrieve("latest RAG benchmark trends")
print([(d.id, d.source, d.score) for d in docs])
```

## 10. Logging

raglib uses Python logging throughout all components.

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
```

Set to `DEBUG` when diagnosing retrieval ranking, routing decisions, retries, and provider fallback behavior.

## 11. Running the Built-in Demo

A full runnable demonstration lives in `raglib/main.py`.

```bash
python -m raglib.main
```

The demo shows:

- corpus setup
- all core components
- multiple RAG strategies
- memory turns
- provider swap and provider chain fallback

## 12. Import Patterns

Direct import from package root:

```python
from raglib import NaiveRAG, Retriever, Generator, MockLLMClient, Document
```

Or explicit import from module path:

```python
from raglib.rag_types.naive_rag import NaiveRAG
```

## 13. Error Handling Patterns

### Provider errors

Wrap web retrieval calls:

```python
from raglib.providers.base_provider import ProviderError

try:
    docs = web_retriever.retrieve("query")
except ProviderError as exc:
    print(f"Web retrieval failed: {exc}")
```

### Empty retrieval fallback

If retrieval returns no documents, you can still generate:

```python
result = generator.generate(query="Explain RAG", documents=[])
print(result.answer)
```

## 14. Production Tips

- Keep document chunks focused and metadata-rich.
- Use `Reranker` and `Evaluator` together for better precision.
- Use `CorrectiveRAG` when recall is variable.
- Use `HybridRAG` for freshness-sensitive use cases.
- Use `MemoryRAG` for multi-turn assistants.
- Keep provider keys out of source code (environment variables or secrets manager).
- Track reasoning traces for observability and debugging.

## 15. Minimal End-to-End Template

```python
from raglib import Document, Retriever, Generator, MockLLMClient, NaiveRAG

docs = [
    Document(id="1", content="RAG combines retrieval and generation."),
    Document(id="2", content="Retriever finds relevant context before answering."),
]

rag = NaiveRAG(
    retriever=Retriever(documents=docs, top_k=2),
    generator=Generator(llm_client=MockLLMClient()),
)

result = rag.run("What is RAG?")
print(result.answer)
```

You now have a complete baseline. From here, swap `NaiveRAG` with any other orchestrator and plug in additional components as needed.
