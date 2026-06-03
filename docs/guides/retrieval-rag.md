# Retrieval (RAG)

The RAG layer lets an agent ground its answers in a document corpus by combining
two complementary kinds of search:

- **Keyword search** (BM25, via [Tantivy](https://github.com/quickwit-oss/tantivy)) — precise
  lexical matching for exact terms, codes, and names.
- **Semantic search** (cosine similarity over embeddings, via
  [ChromaDB](https://www.trychroma.com/)) — recall for paraphrases and related concepts the
  exact words miss.

These two ranked lists are merged with **Reciprocal Rank Fusion (RRF)** and then re-ranked by
semantic similarity. You attach the stores to an [`AgentContext`](../concepts/how-it-works.md)
through its `vector_store` and `keyword_store` fields, or you can drive the search API directly.

Everything in this guide imports from `parsimony_agents.rag`.

## What the RAG layer provides

The layer is built around two session-scoped stores and one fusion function:

| Symbol | Role |
| --- | --- |
| `SessionVectorStore` | Semantic store backed by ChromaDB. Embeds chunks, queries by vector. |
| `SessionKeywordStore` | Full-text store backed by Tantivy. Indexes chunks, queries with BM25. |
| `hybrid_search(...)` | Runs both stores, fuses their rankings with RRF, then re-ranks by cosine similarity. |

`hybrid_search` performs **two stages**:

1. **RRF recall** — each store returns its top candidates; a document's fused score is the sum
   of `1 / (rrf_k + rank)` across the lists it appears in (`rrf_k` defaults to `60`). This
   favours documents that rank well in *either* store, maximising recall.
2. **Semantic re-rank** — the fused candidates are re-scored by cosine similarity between the
   query embedding and each chunk's embedding, then the top `k` are returned. This sharpens
   precision.

The fusion function's full signature:

```python
async def hybrid_search(
    query: str,
    keyword_store: SessionKeywordStore | None,
    vector_store: SessionVectorStore | None,
    identifier: str | None = None,
    k: int = 10,
    rrf_k: int = 60,
) -> list[HybridSearchResult]: ...
```

Either store may be `None`: pass only `keyword_store` for lexical-only search, only
`vector_store` for semantic-only, or both for the full hybrid path. With both `None`, it
returns an empty list.

Each result is a `HybridSearchResult`:

```python
class HybridSearchResult(BaseModel):
    content: str
    metadata: dict
    identifier: str
    rrf_score: float
    rrf_rank: int
    semantic_similarity: float | None = None
```

## Installing the rag extra

Semantic search **requires the `[rag]` extra (chromadb + tantivy)**. Without it,
`SessionVectorStore` raises `ImportError` at construction time because ChromaDB is not present.

```bash
pip install "parsimony-agents[rag]"
```

The extra installs both `chromadb` and `tantivy`. (Tantivy in particular is what backs the
keyword store, so the extra is the supported way to get both halves of hybrid search.) See
[Installation](../getting-started/installation.md) for the other optional extras.

## Session vector store (ChromaDB) and keyword store (Tantivy)

Both stores are **session-scoped** and looked up by `session_id` from a module-level registry.
Use the `get_or_create_session_*` helpers so the same session always resolves to the same store:

```python
from parsimony_agents.rag import (
    get_or_create_session_keyword_store,
    get_or_create_session_vector_store,
)

session_id = "user-session-123"
vector_store = get_or_create_session_vector_store(session_id)
keyword_store = get_or_create_session_keyword_store(session_id)
```

The full set of registry helpers:

| Vector store | Keyword store |
| --- | --- |
| `get_or_create_session_vector_store(session_id)` | `get_or_create_session_keyword_store(session_id)` |
| `get_session_vector_store(session_id)` → store \| `None` | `get_session_keyword_store(session_id)` → store \| `None` |
| `create_session_vector_store(session_id)` | — |
| `cleanup_session_vector_store(session_id)` (async) | `cleanup_session_keyword_store(session_id)` (async) |

### Indexing and querying directly

`SessionVectorStore.index` takes `Document` objects, embeds them, and stores them under an
`identifier`. `query` takes a query embedding (a `list[float]`) and returns `RetrievedChunk`
objects:

```python
import asyncio

from parsimony_agents.rag import (
    Document,
    configure_embeddings,
    embed_query,
    get_or_create_session_vector_store,
)


async def main() -> None:
    configure_embeddings(model="openai/text-embedding-3-small", dimension=1536)

    vector_store = get_or_create_session_vector_store("session-xyz")

    docs = [
        Document(
            content="US unemployment rate reached 3.8% in 2024.",
            metadata={"source": "FRED", "date": "2024-01"},
        ),
        Document(
            content="Employment growth exceeded expectations in Q4.",
            metadata={"source": "Labor Bureau", "date": "2024-12"},
        ),
    ]
    await vector_store.index(docs, identifier="labor_data")

    query_embedding = await embed_query("what is the employment situation?")
    chunks = await vector_store.query(query_embedding, identifier="labor_data", k=5)
    for chunk in chunks:
        print(chunk.score, chunk.content)


if __name__ == "__main__":
    asyncio.run(main())
```

`SessionKeywordStore.index` takes `KeywordDocument` objects and `query` takes a raw query
string (no embedding) — fast, cheap, exact-match recall:

```python
import asyncio

from parsimony_agents.rag import (
    KeywordDocument,
    get_or_create_session_keyword_store,
)


async def main() -> None:
    keyword_store = get_or_create_session_keyword_store("session-xyz")

    docs = [
        KeywordDocument(
            content="US unemployment rate reached 3.8% in 2024.",
            metadata={"source": "FRED"},
            identifier="labor_data",
        ),
    ]
    keyword_store.index(docs, identifier="labor_data")

    results = await keyword_store.query("unemployment 2024", identifier="labor_data", k=10)
    for r in results:
        print(r.score, r.content)


if __name__ == "__main__":
    asyncio.run(main())
```

The `identifier` argument scopes a query to one indexed source (a variable name, a document
group). Omitting it searches across everything indexed in that session.

> **Cleanup is manual.** Stores stay in the module registry, and the keyword store keeps a
> temp directory, until you call `cleanup_session_vector_store(session_id)` /
> `cleanup_session_keyword_store(session_id)` (both async). The vector store cleanup also drops
> its ChromaDB collection.

## Configuring embeddings

Before any embedding call — `embed_texts`, `embed_query`, or anything that touches
`SessionVectorStore` or `hybrid_search` — you must call **`configure_embeddings`** to register
the model and its output dimension. Embeddings run through
[litellm](https://github.com/BerriAI/litellm), so the `model` string is any provider/model
litellm supports.

```python
from parsimony_agents.rag import configure_embeddings

configure_embeddings(
    model="openai/text-embedding-3-small",
    dimension=1536,
    batch_size=100,
)
```

The signature and defaults:

```python
def configure_embeddings(
    *,
    model: str = "gemini/gemini-embedding-2-preview",
    dimension: int,
    batch_size: int = 100,
) -> None: ...
```

`dimension` is required — it has no default. Call `configure_embeddings` once at application
startup, before the first `agent.ask()` / `agent.run()` or any direct store operation. If you
skip it, embeddings fall back to the module default model (`gemini/gemini-embedding-2-preview`)
with no dimension override — almost certainly not what you want, and it requires a Gemini API
key to be present.

The two embedding entry points used by the stores and by `hybrid_search`:

- **`embed_texts(texts, *, task_type="RETRIEVAL_DOCUMENT")`** — embeds a list of corpus
  documents, batched into chunks of `batch_size`. Returns `list[list[float]]`.
- **`embed_query(query)`** — embeds a single search query (with `task_type="RETRIEVAL_QUERY"`).
  Returns `list[float]`.

```python
from parsimony_agents.rag import embed_query, embed_texts

doc_vectors = await embed_texts(["first chunk", "second chunk"])
query_vector = await embed_query("a search question")
```

## Attaching stores to AgentContext

The agent reads its retrieval stores off the `AgentContext` it runs against. **Stores attach
via `AgentContext.vector_store` / `AgentContext.keyword_store`.** Assign the session's stores to
those fields and pass the context to `agent.run()` or `agent.ask()`:

```python
import asyncio

from parsimony_agents import Agent
from parsimony_agents.agent.models import AgentContext
from parsimony_agents.rag import (
    configure_embeddings,
    get_or_create_session_keyword_store,
    get_or_create_session_vector_store,
)


async def main() -> None:
    configure_embeddings(model="openai/text-embedding-3-small", dimension=1536)

    session_id = "user-session-123"
    ctx = AgentContext(session_id=session_id)
    ctx.vector_store = get_or_create_session_vector_store(session_id)
    ctx.keyword_store = get_or_create_session_keyword_store(session_id)

    agent = Agent(model="claude-sonnet-4-6")
    result = await agent.ask(
        "Summarise what the indexed labor data says about unemployment.",
        ctx=ctx,
    )
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
```

Reusing the same `ctx` (with its `session_id`) across calls preserves both the conversation
history and the attached stores — see [Multi-turn conversations](multi-turn.md). If
`vector_store` / `keyword_store` are left `None`, retrieval is simply unavailable for that run.

### Driving hybrid search yourself

You can also run hybrid search outside the agent loop — for example to build a custom retrieval
endpoint or to inspect what the corpus returns:

```python
import asyncio

from parsimony_agents.rag import (
    configure_embeddings,
    get_or_create_session_keyword_store,
    get_or_create_session_vector_store,
    hybrid_search,
)


async def main() -> None:
    configure_embeddings(model="openai/text-embedding-3-small", dimension=1536)

    session_id = "user-session-123"
    keyword_store = get_or_create_session_keyword_store(session_id)
    vector_store = get_or_create_session_vector_store(session_id)

    results = await hybrid_search(
        query="unemployment rate trend",
        keyword_store=keyword_store,
        vector_store=vector_store,
        k=5,
    )
    for r in results:
        sim = "-" if r.semantic_similarity is None else f"{r.semantic_similarity:.3f}"
        print(f"[{r.identifier}] rrf={r.rrf_score:.4f} sim={sim} :: {r.content[:80]}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Document and KeywordDocument models

You hand documents to the two stores using two small Pydantic models — one per store.

**`Document`** is what you index into the vector store. If you omit `id`, a UUID is generated:

```python
class Document(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)
    id: str | None = None
```

A vector-store query returns **`RetrievedChunk`** objects — content plus the cosine-similarity
`score`, the source `identifier`, and the underlying `document_id`:

```python
class RetrievedChunk(BaseModel):
    content: str
    metadata: dict
    score: float
    identifier: str
    document_id: str
```

**`KeywordDocument`** is what you index into the keyword store; it carries its own `identifier`:

```python
class KeywordDocument(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)
    identifier: str = ""
```

A keyword-store query returns **`KeywordSearchResult`** objects, with a BM25 `score`:

```python
class KeywordSearchResult(BaseModel):
    content: str
    metadata: dict
    score: float
    identifier: str
```

`metadata` flows through unchanged from the document you indexed into the result you get back,
so use it to carry source attribution (URL, page number, date) and surface it alongside the
retrieved text.

## See also

- [SQL and document inputs](sql-and-documents.md) — feeding documents and tabular sources into a run.
- [Multi-turn conversations](multi-turn.md) — reusing an `AgentContext` (and its stores) across turns.
- [Configuration](../getting-started/configuration.md) — agent construction options.
- [How it works: the agent loop](../concepts/how-it-works.md) — where `AgentContext` fits.
