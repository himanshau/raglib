# raglib

raglib is a modular, production-grade Python library for Retrieval-Augmented Generation (RAG).

## Installation

Install from PyPI:

pip install raglib-py

Install with optional extras:

pip install raglib-py[tavily,serpapi,faiss]

## Quick import check

python -c "from raglib.rag_types.naive_rag import NaiveRAG; print('Import successful')"

## One-line API quickstart

```python
from raglib import RAG

rag = RAG("my_documents/")
result = rag.query("What is this report about?")

print(result.answer)
print(result.sources)
```

You can also add more sources later:

```python
rag.add("new_data.docx")
rag.add("https://example.com/article")
```

## Features

- Modular retrieval components (vector, web, hybrid, routing, multi-query, multi-hop)
- RAG strategy orchestrators (naive, advanced, corrective, self, agentic, and more)
- Mock LLM client for offline testing
- Provider abstraction with fallback chaining
- Full test suite with unit, integration, e2e, and edge coverage

## Detailed usage guide

See docs/USAGE.md for a full Python usage guide with quickstart, component wiring, and advanced orchestrator examples.

## License

MIT
