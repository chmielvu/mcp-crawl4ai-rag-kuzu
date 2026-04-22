<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I see - since I use MistralAI services, I would use it for both LLM and embeddings. Please check two things for me: is there a graceful degradation already sexisting for agentic rag features? Second: how is reranker handled?

Now I have the full picture on both questions. Here is the precise analysis.

***

## Question 1: Graceful degradation for agentic RAG

**Short answer: partially — the flag exists but the LLM call inside it has no fallback.**

Here is the exact code path from `smart_crawl_url.py`:

```
use_agentic_rag=False  →  code block extraction skipped entirely ✅ clean skip
use_agentic_rag=True   →  calls generate_code_example_summary()
                              └── calls embedding_service.client.chat.completions.create()
                                       └── raises if API key wrong/missing ❌ NO try/except here
```

The outer `for result in all_results` loop has a `try/except Exception as e: continue`  — so a summary generation failure will silently skip that URL's code examples and log an error, but **will not abort the crawl**. The document chunks still get stored. That is accidental graceful degradation — it works, but only because of the broad `continue` in the outer loop, not by design.

The `extract_source_summary()` call (for source metadata) runs **outside** that loop, before it, with no `continue` guard :

```python
# THIS RUNS OUTSIDE THE LOOP — a failure here aborts the whole crawl
for source_id in source_ids:
    source_summary = await crawling_service.extract_source_summary(...)  # ← unguarded
    await database_service.update_source_info(...)
```

This is a real bug. If your Mistral call fails here, the entire `smart_crawl_url` tool returns `{"success": False}` with zero chunks stored.

**Fix — wrap the source summary block:**

```python
# In src/tools/smart_crawl_url.py
# Replace the source_ids loop with:

for source_id in source_ids:
    source_content = "\n\n".join([
        r['markdown'] for r in all_results
        if urlparse(r['url']).netloc == source_id
    ])
    total_word_count = len(source_content.split())

    try:
        source_summary = await crawling_service.extract_source_summary(
            source_id, source_content[:10000]
        )
    except Exception as e:
        logger.warning(f"Summary generation failed for {source_id}, using default: {e}")
        source_summary = f"Content from {source_id}"   # ← same default as crawling.py uses

    await database_service.update_source_info(
        source_id=source_id,
        summary=source_summary,
        word_count=total_word_count
    )
```


***

## Question 2: How the reranker is handled

**Short answer: two separate, inconsistent reranker implementations exist. The one in `perform_rag_query.py` reinstantiates CrossEncoder on every single call.**

### The two implementations

`perform_rag_query.py` has its own inline reranking:

```python
# Instantiates a new CrossEncoder on EVERY tool call — loads model from disk each time
if settings.use_reranking and results:
    model = CrossEncoder(settings.cross_encoder_model)   # ← cold load every request
    reranker = Reranker(model=model, settings=settings)
    reranked = reranker.rerank_results(query, results_dict)
    reranked = reranker.filter_by_threshold(reranked, threshold=settings.default_rerank_threshold)
```

Meanwhile `mcp_server.py` loads the CrossEncoder once at startup into `context.reranking_model` :

```python
if settings.use_reranking:
    reranking_model = CrossEncoder(settings.cross_encoder_model)  # ← loaded once, lives in context
```

But `perform_rag_query.py` **never reads** `context.reranking_model` — it creates its own.  The pre-warmed model in context is passed to `SearchService.rerank_results()` only when called from `search_code_examples.py` (need to verify), but the main RAG query tool ignores it entirely.

**Fix — use the pre-warmed model from context:**

```python
# In src/tools/perform_rag_query.py
# Replace:
#   model = CrossEncoder(settings.cross_encoder_model)
#   reranker = Reranker(model=model, settings=settings)
# With:

if settings.use_reranking and results and context.reranking_model:
    reranker = Reranker(model=context.reranking_model, settings=settings)
    reranked = reranker.rerank_results(query, results_dict)
    reranked = reranker.filter_by_threshold(
        reranked, threshold=settings.default_rerank_threshold
    )
    results_dict = reranked
```


***

## Mistral adapter — both LLM and embeddings

Since you are using Mistral for both, here is the targeted change to `src/services/embeddings.py`. Mistral's embedding API is OpenAI-compatible so the client swap is minimal:

```python
# src/services/embeddings.py  — __init__ method, replace client init block

from crawl4ai_mcp.config import get_settings

class EmbeddingService:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        provider = getattr(self.settings, "llm_provider", "openai")

        if provider == "mistral":
            # Mistral client — pip install mistralai
            from mistralai import Mistral
            self._mistral = Mistral(api_key=self.settings.mistral_api_key)
            self.client = None   # not used for Mistral path
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.settings.openai_api_key)
            self._mistral = None
```

Then replace `create_embedding`:

```python
    async def create_embedding(self, text: str) -> list[float]:
        if self._mistral is not None:
            # mistral-embed: 1024 dims
            response = await self._run_in_executor(
                lambda: self._mistral.embeddings.create(
                    model=self.settings.embedding_model,  # "mistral-embed"
                    inputs=[text],
                )
            )
            return response.data[0].embedding
        # OpenAI path unchanged
        return await self._run_in_executor(
            lambda: self.client.embeddings.create(
                input=text, model=self.settings.embedding_model
            ).data[0].embedding
        )
```

And replace the LLM calls in `crawling.py` (`generate_code_example_summary`, `extract_source_summary`) — both use the same pattern:

```python
# In CrawlingService — replace self.embedding_service.client.chat.completions.create(...)
# with this helper:

def _chat(self, messages: list, max_tokens: int = 150) -> str:
    provider = getattr(self.settings, "llm_provider", "openai")
    if provider == "mistral":
        resp = self.embedding_service._mistral.chat.complete(
            model=self.settings.model_choice,   # e.g. "mistral-small-latest"
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    else:
        resp = self.embedding_service.client.chat.completions.create(
            model=self.settings.model_choice,
            messages=messages,
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
```

Then replace both `await self.embedding_service._run_in_executor(lambda: self.embedding_service.client.chat.completions.create(...))` blocks with `await self.embedding_service._run_in_executor(lambda: self._chat(messages, max_tokens))`.

### Config additions for Mistral

```python
# src/config.py  — add to Settings class
llm_provider: str = Field(default="openai", env="LLM_PROVIDER")
mistral_api_key: str = Field(default="", env="MISTRAL_API_KEY")
```

```bash
# .env
LLM_PROVIDER=mistral
MISTRAL_API_KEY=your-key-here
MODEL_CHOICE=mistral-small-latest
EMBEDDING_MODEL=mistral-embed
EMBEDDING_DIMENSIONS=1024   # mistral-embed output dim

# Also update kuzu_schema.py:
# EMBEDDING_DIM = 1024
```

> ⚠️ **Dimension mismatch warning**: `mistral-embed` outputs **1024 dimensions**, not 1536. If you already have a `.kuzu_db/` from testing with OpenAI embeddings, delete it and re-run `init_db()` — the HNSW vector index dimension is fixed at schema creation time and cannot be altered in place.

