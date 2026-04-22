[PLANS]
- 2026-04-22T13:30:13+02:00 [USER] Execute a phased refactor that removes Supabase and Neo4j in favor of Kuzu, switches provider usage to Mistral AI, refactors reranking, applies related fixes/tweaks, creates a local virtual environment, and verifies the result.
- 2026-04-22T13:30:13+02:00 [CODE] No top-level `plans/` directory exists in this repository; execution is grounded in the `Refactor/` notes plus the current source tree.

[DECISIONS]
- 2026-04-22T13:30:13+02:00 [ASSUMPTION] Treat the existing `pyproject.toml` as the "pproject" the user referenced unless later evidence contradicts that interpretation.
- 2026-04-22T14:10:05+02:00 [CODE] Standardize runtime dependencies around embedded Kuzu storage, Mistral AI embeddings/chat, and FlashRank reranking; remove Supabase and hosted Neo4j assumptions from application flow.
- 2026-04-22T14:10:05+02:00 [CODE] Reuse the server lifespan reranker through `CrawlContext` so reranking is initialized once and shared by tools instead of being cold-created per request.
- 2026-04-22T14:10:05+02:00 [CODE] Scope `mypy` to shipped source modules and ignore missing type data for `crawl4ai`/`flashrank`; test files remain covered by pytest, while production modules remain strictly type-checked.

[PROGRESS]
- 2026-04-22T14:10:05+02:00 [TOOL] Created `.venv` with `uv venv --python 3.12.7 .venv` and synced the project into it with `uv pip install --python .venv -e .`.
- 2026-04-22T14:10:05+02:00 [CODE] Added `src/services/kuzu_schema.py` and `src/services/kuzu_search_backend.py`, then refactored `database.py`, `search.py`, `mcp_server.py`, and tool ingestion helpers to use Kuzu-backed document/code-example storage and retrieval.
- 2026-04-22T14:10:05+02:00 [CODE] Migrated provider-facing code to Mistral in `src/services/embeddings.py` and aligned `src/utilities/text_processing.py` plus crawl summarization flows with the same client path.
- 2026-04-22T14:10:05+02:00 [CODE] Updated docs and environment templates (`README.md`, `.env.example`, `pyproject.toml`) to describe the Kuzu/Mistral/FlashRank stack and local-first setup.

[DISCOVERIES]
- 2026-04-22T13:30:13+02:00 [TOOL] The working tree already contains an untracked `Refactor/` directory; treat it as user-authored input and do not revert it.
- 2026-04-22T14:10:05+02:00 [CODE] Kuzu vector-indexed entities were unreliable when embeddings were populated after node creation; creating `Chunk` and `CodeExample` nodes with embeddings in the initial `CREATE` statement avoids duplicate-key/index issues during ingestion and smoke testing.
- 2026-04-22T14:10:05+02:00 [TOOL] The original `scripts/verify_kuzu.py` was not idempotent because it reused a persistent local database file; it now validates against a temporary database path and closes the connection explicitly.
- 2026-04-22T14:10:05+02:00 [TOOL] `uv run pytest` reports only third-party deprecation warnings from Pydantic/importlib-resources dependencies; no repo-owned test failures or warnings were introduced by the refactor.

[OUTCOMES]
- 2026-04-22T14:10:05+02:00 [TOOL] Verification passed: `uv run mypy src`, `uv run ruff check src scripts`, `uv run pytest`, `uv run python scripts\\verify_kuzu.py`, and `uv build`.
- 2026-04-22T14:10:05+02:00 [CODE] The MCP server now runs without Supabase initialization, stores/searches crawl data in embedded Kuzu, uses Mistral for embeddings/summaries, and uses FlashRank for reranking.
- 2026-04-22T14:10:05+02:00 [CODE] Shared ingestion logic now lives in `src/tools/_ingestion.py`, reducing duplication between `crawl_single_page` and `smart_crawl_url`.
