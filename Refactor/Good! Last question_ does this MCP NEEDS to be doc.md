<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Good! Last question: does this MCP NEEDS to be dockerized? Or with the changes (Kuzu et c.) it  can be just run locally as stadio/HTTP MCP?

Complete picture. Here is the precise answer.

***

## No, Docker is not required

The code fully supports local execution. `run_server()` checks `settings.transport` and branches to either `mcp.run_stdio_async()` or `mcp.run_sse_async()` — both are native `FastMCP` methods that need nothing but a Python process.  Docker exists purely as a convenience packaging option.

The actual blocker to running locally is not Docker — it is **Supabase**. The `get_supabase_client()` call in the lifespan is unconditional:

```python
# This runs on EVERY startup regardless of settings
supabase_client = get_supabase_client()   # ← hard crash if SUPABASE_URL is empty
```

With your Kuzu substitution from the original scope, this call disappears and the only external dependency becomes the Playwright/Chromium browser that Crawl4AI manages itself (`crawl4ai-setup` installs it).

***

## Full local setup after all our changes

### 1. Install

```bash
git clone https://github.com/Wirasm/mcp-crawl4ai-rag
cd mcp-crawl4ai-rag

# uv is already declared in the repo
uv venv && source .venv/bin/activate

# Install deps (with our substitutions)
uv pip install -e ".[dev]"
uv pip install flashrank mistralai kuzu
uv pip uninstall sentence-transformers  # if only used for reranking

# Install Playwright chromium (one-time, ~170MB)
crawl4ai-setup
```


### 2. `.env` for fully local operation

```bash
# Transport — stdio for Claude Desktop/Code, sse for HTTP
TRANSPORT=stdio          # or: sse
HOST=127.0.0.1
PORT=8051

# Mistral — LLM + embeddings
LLM_PROVIDER=mistral
MISTRAL_API_KEY=your-key
MODEL_CHOICE=mistral-small-latest
EMBEDDING_MODEL=mistral-embed
EMBEDDING_DIMENSIONS=1024

# Local Kuzu DB (replaces Supabase entirely)
KUZU_DB_PATH=./data/kuzu_db

# Reranker — local ONNX, no API key
USE_RERANKING=true
CROSS_ENCODER_MODEL=ms-marco-MiniLM-L-12-v2
RERANKER_CACHE_DIR=./data/flashrank_cache
RERANKER_MAX_LENGTH=512
DEFAULT_RERANK_THRESHOLD=0.3

# Feature flags
USE_CONTEXTUAL_EMBEDDINGS=false   # set true if you want LLM-augmented chunk context
USE_AGENTIC_RAG=true
USE_HALLUCINATION_DETECTION=false # was Neo4j-backed, skip until re-implemented

# Supabase — set to dummy values so pydantic-settings doesn't error on missing fields
# (until you remove the supabase dependency from config.py)
SUPABASE_URL=http://localhost
SUPABASE_SERVICE_KEY=dummy
```

> **Note on `SUPABASE_*` dummies**: `config.py` declares these as required `str` fields with no default.  Until you remove them from the Settings model, pydantic-settings will raise `ValidationError` at import time if they are absent. Setting them to dummy strings is the zero-touch fix; the proper fix is to delete those fields from `config.py` after the Kuzu migration.

### 3. Run

```bash
# stdio (for Claude Desktop / Claude Code)
TRANSPORT=stdio python -m src.main

# SSE/HTTP (for any HTTP MCP client, Postman testing, etc.)
TRANSPORT=sse python -m src.main
# → server listens on http://127.0.0.1:8051/sse
```


### 4. Claude Desktop config (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "src.main"],
      "cwd": "/path/to/mcp-crawl4ai-rag",
      "env": {
        "TRANSPORT": "stdio",
        "MISTRAL_API_KEY": "your-key",
        "KUZU_DB_PATH": "./data/kuzu_db"
      }
    }
  }
}
```

For **Claude Code** (VS Code extension), the same JSON block goes into `.vscode/mcp.json` or workspace settings under `"mcp.servers"` — same shape, same fields.

### 5. What still needs a network call at runtime

| Component | Local? | Notes |
| :-- | :-- | :-- |
| Chromium browser | ✅ Local | Playwright-managed, installed by `crawl4ai-setup` |
| Kuzu vector DB | ✅ Local | Embedded, file-based |
| FlashRank reranker | ✅ Local | ONNX, downloaded once to `RERANKER_CACHE_DIR` |
| Mistral embeddings | ☁️ API | `mistral-embed` call per chunk on ingest |
| Mistral LLM | ☁️ API | Only when `USE_AGENTIC_RAG=true` or `USE_CONTEXTUAL_EMBEDDINGS=true` |
| Supabase | ❌ Removed | After Kuzu migration |

The only unavoidable network dependency is Mistral — which you already have. Everything else runs in-process.

