# raglib-py

raglib-py is a production-grade Retrieval-Augmented Generation library for Python.

This README is designed to be the full user guide for PyPI users, so you do not need to go anywhere else to start.

Important package note:

- PyPI package name: raglib-py
- Python import name: raglib

## Current Support Counts

- Implemented RAG strategies: 12
- Built-in chat LLM providers: 5
- Custom chat model support: yes (BaseLLMClient or LangChain-style invoke model)

The 5 built-in chat LLM providers are:

- openai
- anthropic
- groq
- google
- ollama

## What You Can Do

- Load documents from files, folders, URLs, or raw text
- Use one clean entry point: RAG(...)
- Switch between local and cloud LLM providers
- Choose embedding provider and vector store backend
- Run 12 RAG strategies out of the box
- Use interactive terminal chat mode

## Installation

Install core package:

```bash
pip install raglib-py
```

Common optional extras:

```bash
# Local Ollama chat/embedding support
pip install "raglib-py[ollama]"

# Chroma vector DB backend
pip install "raglib-py[chroma]"

# DOCX/PDF/PPTX document loading
pip install "raglib-py[docx,pdf,pptx]"

# All major runtime extras
pip install "raglib-py[all]"
```

Quick import check:

```bash
python -c "from raglib import RAG; print('OK')"
```

## Quickstart (Zero API Keys)

```python
from raglib import RAG

rag = RAG("RAG improves grounded generation using retrieved context.")
result = rag.query("What does RAG improve?")

print(result.answer)
print([doc.id for doc in result.sources])
```

In this mode, raglib uses offline defaults automatically:

- chat model: MockLLMClient
- embeddings: MockEmbedding

## Real Local Stack (Ollama)

If you have local models in Ollama, this is a strong default setup:

```python
from raglib import RAG

rag = RAG(
	source=r"C:\path\to\your\document.docx",
	chat_llm="ollama",
	chat_model="gemma3:4b",
	embedding_llm="ollama",
	embedding_model="nomic-embed-text:latest",
	vector_db="chroma",
	rag_type="corrective",
	top_k=5,
)

result = rag.query("What is the core concept of this paper?")
print(result.answer)
```

## Cloud Chat + Local Embeddings Example

```python
from raglib import RAG

rag = RAG(
	source="Service test content.",
	chat_llm="groq",
	chat_api_key="YOUR_GROQ_API_KEY",
	chat_model="qwen/qwen3-32b",
	embedding_llm="ollama",
	embedding_model="nomic-embed-text:latest",
	vector_db="memory",
	rag_type="naive",
)

print(rag.query("Reply in one line that service is available.").answer)
```

## Supported Input Sources

source accepts:

- File path: .txt, .md, .docx, .pptx, .pdf
- Folder path: recursive load
- URL: web page text extraction
- Raw text string
- List of any mix of the above

You can ingest more data later:

```python
rag.add("new_notes.md")
rag.add("https://example.com/post")
```

## RAG API (Main Constructor)

```python
RAG(
	source=None,
	chat_llm=None,
	embedding_llm=None,
	vision_llm=None,
	llm_key=None,
	chat_api_key=None,
	embedding_api_key=None,
	vision_api_key=None,
	rag_type="corrective",
	top_k=5,
	chunk_size=400,
	chunk_overlap=50,
	output_dir=None,
	chat_model=None,
	chat_base_url=None,
	embedding_model=None,
	embedding_base_url=None,
	vision_model=None,
	vision_base_url=None,
	vector_db=None,
	vector_db_kwargs=None,
)
```

## API Keys And Endpoints (Important)

raglib never provides API keys. You must use your own provider credentials.

Use these fields in `RAG(...)`:

- `chat_api_key`: key for `chat_llm` provider
- `embedding_api_key`: key for `embedding_llm` provider
- `vision_api_key`: key for `vision_llm` provider
- `llm_key`: one shared fallback key when you do not want to pass separate keys

Endpoint fields:

- `chat_base_url`: custom OpenAI-compatible chat endpoint
- `embedding_base_url`: custom Ollama embedding endpoint
- `vision_base_url`: custom OpenAI-compatible vision endpoint

Provider key mapping:

- `chat_llm="openai" | "anthropic" | "groq" | "google"` needs a chat key
- `embedding_llm="openai" | "google"` needs an embedding key
- `vision_llm="openai" | "anthropic" | "google"` needs a vision key
- `ollama`, `mock`, and local `huggingface` modes do not require cloud API keys

Example with separate keys:

```python
from raglib import RAG

rag = RAG(
	source="docs/",
	chat_llm="openai",
	chat_api_key="YOUR_OPENAI_CHAT_KEY",
	embedding_llm="google",
	embedding_api_key="YOUR_GOOGLE_KEY",
	vision_llm="anthropic",
	vision_api_key="YOUR_ANTHROPIC_KEY",
)
```

Example with one shared key:

```python
rag = RAG(
	source="docs/",
	chat_llm="openai",
	embedding_llm="openai",
	vision_llm="openai",
	llm_key="YOUR_OPENAI_KEY",
)
```

Key methods:

- query(question): ask one question and get GenerationResult
- add(source): add more documents to existing index
- chat(): start terminal interactive Q/A session

## How Many RAG Strategies Are Included?

raglib currently provides 12 built-in RAG strategies:

1. naive
2. advanced
3. corrective
4. self
5. agentic
6. hybrid
7. multi_query
8. multi_hop
9. routing
10. memory
11. web
12. tool

Use them by setting rag_type:

```python
rag = RAG(source="docs/", rag_type="multi_hop")
```

When to use what:

- naive: fastest baseline
- advanced: better quality via rerank/reduce/dedup
- corrective: retries when context quality is weak
- self: decision + reflection-driven behavior
- agentic: planner-based sub-query execution
- hybrid: local + web retrieval blending
- multi_query: query variant expansion
- multi_hop: multi-step reasoning retrieval
- routing: automatic retrieval route selection
- memory: conversation-memory-aware answering
- web: web-first retrieval
- tool: retrieval plus tool output injection

## Provider Support

Chat providers (5 built-in + custom adapter support):

- openai
- anthropic
- groq
- google
- ollama
- custom BaseLLMClient or LangChain-style invoke() model

Embedding providers:

- openai
- google
- ollama
- huggingface (aliases: free, local)
- mock

Vision providers (for scanned PDF fallback):

- openai
- anthropic
- google
- mock

Vector backends:

- chroma (default selection)
- memory
- custom BaseVectorStore instance

## Interactive Terminal Chat

```python
from raglib import RAG

rag = RAG("my_docs/")
rag.chat()
```

Commands inside session:

- help
- history
- clear
- exit / quit / q

## Output Saving

Set output_dir to save each query result as JSON:

```python
rag = RAG(source="docs/", output_dir="outputs")
rag.query("Summarize this")
```

## Troubleshooting

Import confusion:

- Install: pip install raglib-py
- Import: from raglib import RAG

Chroma issues:

- If Chroma is unavailable or unstable, use vector_db="memory"

Missing provider package:

- Install needed extra (for example: pip install "raglib-py[groq]")

## Complete Local Test Script

A ready user script is included at:

- examples/local_ollama_chroma_test.py

An all-strategy comparison script is included at:

- examples/check_all_rag_types.py

## License

MIT
