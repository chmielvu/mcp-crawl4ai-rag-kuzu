<h1 align="center">Crawl4AI RAG MCP Server</h1>

<p align="center">
  <em>Web Crawling and Graph RAG Capabilities for AI Agents and AI Coding Assistants</em>
</p>

A high-performance implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) providing web crawling and Graph RAG capabilities over a modern, remote service architecture:

- **Remote Crawl4AI 0.8.6 REST Service**: Delegated web crawling via HTTP adapter.
- **Remote FalkorDB Graph Database**: Dedicated `crawl-graph` storing graph topology, vector embeddings, and full-text indexes.
- **Unified-ML Microservice**: 384-dimensional multilingual embeddings (`intfloat/multilingual-e5-small`), cross-encoder reranking (`ms-marco-MultiBERT-L-12`), and zero-shot GLiNER entity/relation extraction (`fastino/gliner2-multi-v1`).
- **Mistral AI**: Chat-only completions and summaries (`mistral-small-latest`).
- **Optional LangExtract Enrichment**: Grounded page-level structured metadata extraction with character span provenance.

---

## Architecture & Service Boundaries

```
┌───────────────────────────────────────────────────────────────────┐
│                           MCP Clients                             │
│               (Claude Desktop, Windsurf, Codex, etc.)             │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │ (stdio / SSE)
┌─────────────────────────────────▼─────────────────────────────────┐
│                      Crawl4AI RAG MCP Server                      │
│                                                                   │
│  ┌─────────────────────────┐       ┌───────────────────────────┐  │
│  │   Crawl4AI REST Client  │       │     Unified-ML Client     │  │
│  │ (Remote Crawl4AI 0.8.6) │       │   (Embed / Rerank / GLiNER│  │
│  └────────────┬────────────┘       └─────────────┬─────────────┘  │
│               │                                  │                │
│  ┌────────────▼────────────┐       ┌─────────────▼─────────────┐  │
│  │     FalkorDB Store      │       │    Mistral Chat Adapter   │  │
│  │  (`crawl-graph` store)  │       │  (Summaries & Completions)│  │
│  └─────────────────────────┘       └───────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

### 1. Remote Crawl4AI 0.8.6 REST Adapter
Web crawling is handled by a remote Crawl4AI 0.8.6 HTTP service configured via `CRAWL4AI_BASE_URL` (and optional `CRAWL4AI_API_TOKEN`). All single-page and batch crawls are dispatched to the remote `/crawl` endpoint with configurable concurrency (`CRAWL4AI_MAX_BATCH_SIZE`) and timeouts (`CRAWL4AI_TIMEOUT_SECONDS`).

### 2. Dedicated FalkorDB Graph Storage (`crawl-graph`)
Graph data is persisted in a remote FalkorDB instance (`FALKORDB_URL`) under the dedicated graph name `crawl-graph` (`FALKORDB_GRAPH=crawl-graph`).
- **Nodes**: `Site`, `CrawlRun`, `Page`, `Chunk`, `__Entity__`.
- **Relationships**: `(:Site)-[:HAS_PAGE]->(:Page)`, `(:CrawlRun)-[:CRAWLED]->(:Page)`, `(:Page)-[:HAS_CHUNK]->(:Chunk)`, `(:Page)-[:LINKS_TO]->(:Page)`, `(:Site)-[:HAS_ENTITY]->(:__Entity__)`, `(:__Entity__)-[:MENTIONED_IN]->(:Chunk)`, and `(:__Entity__)-[:RELATES]->(:__Entity__)`.
- **Indexes**: 384-dimensional vector indexes for semantic chunk search and entity search, plus full-text search indexes.
- **Connection Management**: Configurable query timeout (`FALKORDB_QUERY_TIMEOUT_MS`) and connection pooling (`FALKORDB_MAX_CONNECTIONS`).

### 3. Unified-ML Service
All dense vector embeddings, reranking, and zero-shot entity extraction are offloaded to a Unified-ML microservice (`UNIFIED_ML_BASE_URL`):
- **Embeddings**: 384-dimensional embeddings generated with `intfloat/multilingual-e5-small`. Follows asymmetric search conventions with required `passage: ` and `query: ` prefixes.
- **Reranking**: Candidate reranking powered by `ms-marco-MultiBERT-L-12`.
- **GLiNER Extraction**: Zero-shot named entity recognition and relation extraction powered by `fastino/gliner2-multi-v1` (`USE_GLINER_METADATA=true`), extracting entities and relationship triples directly from crawled text.

### 4. Mistral AI (Chat & Summaries Only)
Mistral AI (`MISTRAL_API_KEY`, `MODEL_CHOICE=mistral-small-latest`) is used strictly for chat completions:
- Generating concise summaries of extracted code examples.
- Generating high-level documentation summaries for indexed sources.
- No embedding models are requested from Mistral AI.

### 5. Optional LangExtract Metadata Enrichment
When `USE_LANGEXTRACT_METADATA=true`, page-level grounded extractions are produced via the `langextract` library using Mistral to discover structured attributes and character interval spans (`start_char`, `end_char`).

---

## Fresh Crawl Requirement

The graph schema is initialized idempotently on the dedicated `crawl-graph` in FalkorDB. Because the architecture uses a native graph model with 384-dimensional vector embeddings, a **fresh crawl is required** to populate the database. There is no legacy migration utility.

---

## Typed MCP Tools

The server exposes five typed MCP tools. FastMCP serializes each Pydantic response as structured data; the tools do not return JSON strings.

### 1. `crawl_single_page`
Crawls one URL through remote Crawl4AI, performs site-level GLiNER enrichment, chunks and embeds the page, and writes one atomic crawl payload to FalkorDB.

- **Parameters**: `url` (`str`)
- **Response**: `SingleCrawlResponse` with `success`, `url`, `run_id`, `pages_crawled`, `chunks_stored`, structured `failures`, optional `error`, and `message`.

### 2. `smart_crawl_url`
Handles recursive internal-link crawling, XML sitemaps, and text/markdown files.

- **Parameters**: `url` (`str`), `max_depth` (`int`, default `3`), `max_concurrent` (`int`, default `10`), `chunk_size` (`int`, default `5000`), and `timeout` (`int`, default `300`).
- **Response**: `SmartCrawlResponse` with `success`, `url`, `crawl_type`, `run_id`, `urls_processed`, `pages_crawled`, `chunks_stored`, structured `failures`, optional `error`, and `message`.

### 3. `get_available_sites`
Lists indexed sites with page/chunk counts and persisted GLiNER metadata.

- **Parameters**: None.
- **Response**: `AvailableSitesResponse` with `success`, `sites`, `total_sites`, optional `error`, and `message`. Each `SiteInfo` contains `site_id`, `domain`, `root_url`, `summary`, `first_seen`, `last_crawled`, `page_count`, `chunk_count`, and `gliner_metadata`.

### 4. `perform_rag_query`
Performs semantic or hybrid chunk retrieval with optional Unified-ML reranking and entity/relation provenance expansion.

- **Parameters**: `query` (`str`), `source` (`str | None`, default `None`), `match_count` (`int`, default `5`), and optional `use_hybrid`/`use_reranking` flags.
- **Response**: `RagSearchResponse` with typed `SearchHit` results and structured failure details.

### 5. `search_code_examples`
Searches shared `Chunk` nodes filtered by `content_type="code"` and optional language/site filters.

- **Parameters**: `query` (`str`), `source_id` (`str | None`, default `None`), `language` (`str | None`, default `None`), `match_count` (`int`, default `5`), and optional `use_reranking`.
- **Response**: `CodeSearchResponse` with typed `SearchHit` results and structured failure details.

---

## Configuration

Copy `.env.example` to `.env` and configure your environment:

```bash
cp .env.example .env
```

### Configuration Reference

| Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **Server** | | | |
| `HOST` | `str` | `0.0.0.0` | MCP server listen host |
| `PORT` | `int` | `8051` | MCP server listen port |
| `TRANSPORT` | `str` | `sse` | Transport type (`sse` or `stdio`) |
| `PYTHONUNBUFFERED` | `int` | `1` | Disables output buffering |
| `MCP_PROJECT_ROOT` | `str` | _auto-detected_ | Absolute path to project root |
| **Mistral AI** | | | |
| `MISTRAL_API_KEY` | `str` | _required_ | Mistral API key for chat/summaries |
| `MODEL_CHOICE` | `str` | `mistral-small-latest` | Chat completion model |
| **Remote Crawl4AI** | | | |
| `CRAWL4AI_BASE_URL` | `str` | `http://localhost:11235` | Remote Crawl4AI REST service endpoint |
| `CRAWL4AI_API_TOKEN` | `str` | `""` | Optional Bearer token for Crawl4AI service |
| `CRAWL4AI_TIMEOUT_SECONDS` | `float`| `60.0` | HTTP timeout for crawl requests |
| `CRAWL4AI_MAX_BATCH_SIZE` | `int` | `100` | Maximum URLs per batch request (1–100) |
| **FalkorDB** | | | |
| `FALKORDB_URL` | `str` | `falkor://localhost:6380` | FalkorDB connection URL |
| `FALKORDB_GRAPH` | `str` | `crawl-graph` | Dedicated graph name (must be `crawl-graph`) |
| `FALKORDB_QUERY_TIMEOUT_MS` | `int` | `1000` | Query timeout in milliseconds |
| `FALKORDB_MAX_CONNECTIONS` | `int` | `16` | Connection pool size |
| **Unified-ML** | | | |
| `UNIFIED_ML_BASE_URL` | `str` | `http://localhost:8000` | Unified-ML microservice endpoint |
| `UNIFIED_ML_EMBED_MODEL` | `str` | `intfloat/multilingual-e5-small` | Deployed embedding model |
| `UNIFIED_ML_EMBEDDING_DIMENSIONS` | `int` | `384` | Embedding dimension (must be `384`) |
| `UNIFIED_ML_TIMEOUT_SECONDS` | `float`| `30.0` | HTTP timeout for ML service |
| `UNIFIED_ML_BATCH_SIZE` | `int` | `32` | Batch size for embedding calls |
| **GLiNER Extraction** | | | |
| `USE_GLINER_METADATA` | `bool` | `true` | Enable zero-shot entity/relation extraction |
| `GLINER_ENTITY_LABELS` | `str` | `product,technology,library,organization,person` | Comma-separated entity labels |
| `GLINER_RELATION_LABELS` | `str` | `uses,depends_on,implements,stores` | Comma-separated relation labels |
| `GLINER_THRESHOLD` | `float`| `0.5` | Confidence threshold for GLiNER facts |
| `GLINER_INCLUDE_CONFIDENCE` | `bool` | `true` | Store confidence scores in graph |
| `GLINER_INCLUDE_SPANS` | `bool` | `true` | Store character spans in graph |
| **LangExtract (Optional)** | | | |
| `USE_LANGEXTRACT_METADATA` | `bool` | `false` | Enable page-level LangExtract enrichment |
| `LANGEXTRACT_MODEL_ID` | `str` | `mistral-small-latest` | Model ID for LangExtract |
| `LANGEXTRACT_BASE_URL` | `str` | `https://api.mistral.ai/v1` | Mistral API endpoint for LangExtract |
| `LANGEXTRACT_EXTRACTION_PASSES` | `int` | `1` | Extraction passes per page |
| `LANGEXTRACT_MAX_WORKERS` | `int` | `4` | Concurrency for LangExtract workers |
| `LANGEXTRACT_MAX_CHAR_BUFFER` | `int` | `2000` | Character chunk buffer size |
| **Strategy Flags** | | | |
| `USE_CONTEXTUAL_EMBEDDINGS` | `bool` | `false` | Generate LLM context prefix before embedding |
| `USE_HYBRID_SEARCH` | `bool` | `false` | Combine vector ANN with full-text search |
| `USE_RERANKING` | `bool` | `false` | Apply Unified-ML cross-encoder reranking |
| `USE_AGENTIC_RAG` | `bool` | `true` | Extract and index code examples |

---

## Setup & Installation

### Prerequisites
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- Running **FalkorDB** instance (e.g. `docker run -p 6380:6379 -it --rm falkordb/falkordb`)
- Running **Unified-ML** service (providing `/health`, `/info`, `/embeddings`, `/rerank`, `/extract`)
- Running **Crawl4AI 0.8.6** REST service (providing `/health`, `/crawl`)
- A valid **Mistral AI** API key

### 1. Install Dependencies
```bash
uv venv .venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### 2. Configure Environment
Create a `.env` file from `.env.example` and supply your `MISTRAL_API_KEY` along with remote service URLs.

### 3. Verify Remote Services Contract
Run the remote service contract verification script to validate connectivity and model contracts before starting the MCP server:

```bash
uv run python scripts/verify_remote_services.py
```

> **Note**: Live verification requires active, reachable Crawl4AI and Unified-ML services. The diagnostic script verifies health status, model names, 384-dimensional embeddings, reranking, and GLiNER extraction against the remote endpoints.

### 4. Start MCP Server
```bash
uv run crawl4ai-mcp
```

---

## Structured Failure Behavior

The server is built for predictable failure isolation across all remote dependencies:

- **Startup Validation**: Settings validate required credentials and fail fast if removed legacy variables are present in the environment.
- **Provider Error Isolation**: Remote HTTP timeouts and status errors are wrapped into structured exception types (`Crawl4AIProviderError`, `UnifiedMLProviderError`, `ChatProviderError`).
- **Resilient Batch Ingestion**: In batch crawls, individual URL fetch failures return structured `CrawlFailure` entries without aborting successful page ingests.
- **JSON Error Envelopes**: All MCP tools return structured JSON payloads with `"success": false` and clear `"error"` messages upon failure.

---

## MCP Client Configuration

### SSE Transport (Default)
Connect your MCP client to the running HTTP SSE server:

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "sse",
      "url": "http://localhost:8051/sse"
    }
  }
}
```

*(For Windsurf, use `"serverUrl": "http://localhost:8051/sse"`)*

### Stdio Transport
For direct process invocation via MCP clients (Claude Desktop, Windsurf, Codex):

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/mcp-crawl4ai-rag", "run", "crawl4ai-mcp"],
      "env": {
        "TRANSPORT": "stdio",
        "PYTHONUNBUFFERED": "1",
        "MCP_PROJECT_ROOT": "/absolute/path/to/mcp-crawl4ai-rag",
        "MISTRAL_API_KEY": "your_mistral_api_key",
        "CRAWL4AI_BASE_URL": "http://localhost:11235",
        "FALKORDB_URL": "falkor://localhost:6380",
        "FALKORDB_GRAPH": "crawl-graph",
        "UNIFIED_ML_BASE_URL": "http://localhost:8000",
        "UNIFIED_ML_EMBED_MODEL": "intfloat/multilingual-e5-small",
        "UNIFIED_ML_EMBEDDING_DIMENSIONS": "384"
      }
    }
  }
}
```
