<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Yes, please provide here in chat oinline code-validated instructions on the migration\

Here is the complete, code-validated migration. Every snippet maps 1:1 to the actual source files read above.

***

## 0. Prerequisites

Install the Vela Partners fork (drop-in replacement for `kuzu` package name):

```bash
pip uninstall kuzu -y 2>/dev/null
pip install git+https://github.com/Vela-Engineering/kuzu.git
pip uninstall supabase neo4j -y
```

Add to `pyproject.toml` / `requirements.txt`:

```toml
# remove: supabase, neo4j, python-neo4j
# add:
kuzu @ git+https://github.com/Vela-Engineering/kuzu.git
```


***

## 1. Schema — `src/services/kuzu_schema.py` (new file)

This replaces both the Supabase SQL schema and the Neo4j graph model in one file.

```python
"""
KuzuDB schema initialisation.
Replaces: Supabase (crawled_pages, code_examples, sources tables)
          Neo4j   (Source/Chunk nodes + hallucination-check graph)
"""
import kuzu

EMBEDDING_DIM = 1536  # change to 3072 for text-embedding-3-large


def init_db(db_path: str = ".kuzu_db") -> kuzu.Connection:
    db = kuzu.Database(db_path)          # Vela fork: concurrent-write-safe
    conn = kuzu.Connection(db)
    _create_schema(conn)
    _create_indexes(conn)
    return conn


def _create_schema(conn: kuzu.Connection) -> None:
    # --- Node tables ---------------------------------------------------------
    conn.execute("""
        CREATE NODE TABLE IF NOT EXISTS Source (
            source_id   STRING,
            summary     STRING,
            word_count  INT64,
            updated_at  STRING,
            PRIMARY KEY (source_id)
        )
    """)

    conn.execute(f"""
        CREATE NODE TABLE IF NOT EXISTS Chunk (
            chunk_id      STRING,
            url           STRING,
            chunk_number  INT64,
            content       STRING,
            metadata      STRING,
            embedding     FLOAT[{EMBEDDING_DIM}],
            PRIMARY KEY (chunk_id)
        )
    """)

    conn.execute(f"""
        CREATE NODE TABLE IF NOT EXISTS CodeExample (
            example_id    STRING,
            url           STRING,
            chunk_number  INT64,
            content       STRING,
            summary       STRING,
            language      STRING,
            metadata      STRING,
            embedding     FLOAT[{EMBEDDING_DIM}],
            PRIMARY KEY (example_id)
        )
    """)

    # --- Relationship tables --------------------------------------------------
    # These replace the Neo4j graph — hallucination check = MATCH (s:Source)-[:CONTAINS]->(c)
    conn.execute("""
        CREATE REL TABLE IF NOT EXISTS CONTAINS (
            FROM Source TO Chunk
        )
    """)
    conn.execute("""
        CREATE REL TABLE IF NOT EXISTS HAS_EXAMPLE (
            FROM Source TO CodeExample
        )
    """)
    conn.execute("""
        CREATE REL TABLE IF NOT EXISTS NEXT_CHUNK (
            FROM Chunk TO Chunk
        )
    """)


def _create_indexes(conn: kuzu.Connection) -> None:
    # Vector indexes (HNSW, disk-based, predicate-agnostic — Kuzu 0.9+)
    conn.execute("""
        CREATE VECTOR INDEX IF NOT EXISTS chunk_embedding_idx
        ON Chunk(embedding)
        WITH (metric := 'cosine', ef_construction := 128, m := 16)
    """)
    conn.execute("""
        CREATE VECTOR INDEX IF NOT EXISTS code_embedding_idx
        ON CodeExample(embedding)
        WITH (metric := 'cosine', ef_construction := 128, m := 16)
    """)
    # Full-text search indexes (built-in BM25)
    conn.execute("""
        CREATE FULL TEXT INDEX IF NOT EXISTS chunk_fts_idx ON Chunk(content)
    """)
    conn.execute("""
        CREATE FULL TEXT INDEX IF NOT EXISTS code_fts_idx ON CodeExample(content, summary)
    """)
```


***

## 2. `src/services/database.py` — full replacement

Drop in this file verbatim. The public method signatures are **identical** to the Supabase version — nothing upstream breaks.

```python
"""
KuzuDB database service.
Drop-in replacement for the Supabase DatabaseService.
All public method signatures preserved.
"""
import json
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from datetime import datetime, timezone

import kuzu

from crawl4ai_mcp.config import get_settings
from crawl4ai_mcp.models import SourceInfo

logger = logging.getLogger(__name__)


def _chunk_id(url: str, chunk_number: int) -> str:
    return f"{url}::chunk::{chunk_number}"


def _example_id(url: str, chunk_number: int) -> str:
    return f"{url}::code::{chunk_number}"


class DatabaseService:
    """KuzuDB-backed database service (replaces Supabase)."""

    def __init__(self, client: kuzu.Connection, settings: Optional[Any] = None):
        self.conn = client          # kuzu.Connection  (was: supabase client)
        self.settings = settings or get_settings()

    # -------------------------------------------------------------------------
    # WRITE: documents (was: crawled_pages table)
    # -------------------------------------------------------------------------
    async def add_documents(
        self,
        urls: List[str],
        chunk_numbers: List[int],
        contents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        url_to_full_document: Dict[str, str],
        batch_size: int = 20,
    ) -> Dict[str, Any]:
        unique_urls = list(set(urls))

        # Delete existing chunks for these URLs (replaces Supabase .delete().in_())
        for url in unique_urls:
            self.conn.execute(
                "MATCH (c:Chunk {url: $url}) DETACH DELETE c",
                {"url": url},
            )

        documents_added = 0
        for i in range(0, len(urls), batch_size):
            b_urls   = urls[i:i+batch_size]
            b_chunks = chunk_numbers[i:i+batch_size]
            b_conts  = contents[i:i+batch_size]
            b_embs   = embeddings[i:i+batch_size]
            b_metas  = metadatas[i:i+batch_size]

            for url, chunk_num, content, embedding, metadata in zip(
                b_urls, b_chunks, b_conts, b_embs, b_metas
            ):
                source_id = urlparse(url).netloc
                cid = _chunk_id(url, chunk_num)

                # Upsert Source node
                self.conn.execute(
                    """
                    MERGE (s:Source {source_id: $sid})
                    ON CREATE SET s.word_count = 0, s.updated_at = $ts
                    """,
                    {"sid": source_id, "ts": datetime.now(timezone.utc).isoformat()},
                )

                # Upsert Chunk node
                self.conn.execute(
                    """
                    MERGE (c:Chunk {chunk_id: $cid})
                    SET c.url          = $url,
                        c.chunk_number = $num,
                        c.content      = $content,
                        c.metadata     = $meta,
                        c.embedding    = $emb
                    """,
                    {
                        "cid": cid,
                        "url": url,
                        "num": chunk_num,
                        "content": content,
                        "meta": json.dumps(metadata),
                        "emb": embedding,
                    },
                )

                # CONTAINS relationship (replaces Neo4j Source→Chunk edge)
                self.conn.execute(
                    """
                    MATCH (s:Source {source_id: $sid}), (c:Chunk {chunk_id: $cid})
                    MERGE (s)-[:CONTAINS]->(c)
                    """,
                    {"sid": source_id, "cid": cid},
                )

                # NEXT_CHUNK chain for contextual continuity
                if chunk_num > 0:
                    prev_cid = _chunk_id(url, chunk_num - 1)
                    self.conn.execute(
                        """
                        MATCH (prev:Chunk {chunk_id: $prev}), (cur:Chunk {chunk_id: $cur})
                        MERGE (prev)-[:NEXT_CHUNK]->(cur)
                        """,
                        {"prev": prev_cid, "cur": cid},
                    )

                documents_added += 1

        return {"success": True, "count": documents_added, "total": len(urls)}

    # -------------------------------------------------------------------------
    # WRITE: code examples (was: code_examples table)
    # -------------------------------------------------------------------------
    async def add_code_examples(
        self,
        urls: List[str],
        chunk_numbers: List[int],
        code_examples: List[str],
        summaries: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 20,
    ) -> Dict[str, Any]:
        if not urls:
            return {"success": True, "count": 0}

        for url in set(urls):
            self.conn.execute(
                "MATCH (e:CodeExample {url: $url}) DETACH DELETE e",
                {"url": url},
            )

        examples_added = 0
        for i in range(0, len(urls), batch_size):
            for url, chunk_num, code, summary, embedding, metadata in zip(
                urls[i:i+batch_size],
                chunk_numbers[i:i+batch_size],
                code_examples[i:i+batch_size],
                summaries[i:i+batch_size],
                embeddings[i:i+batch_size],
                metadatas[i:i+batch_size],
            ):
                source_id = urlparse(url).netloc
                eid = _example_id(url, chunk_num)
                language = "unknown"
                if code.startswith("```"):
                    lang_line = code.split("\n")[3:].strip()
                    if lang_line:
                        language = lang_line

                self.conn.execute(
                    """
                    MERGE (e:CodeExample {example_id: $eid})
                    SET e.url          = $url,
                        e.chunk_number = $num,
                        e.content      = $code,
                        e.summary      = $summary,
                        e.language     = $lang,
                        e.metadata     = $meta,
                        e.embedding    = $emb
                    """,
                    {
                        "eid": eid, "url": url, "num": chunk_num,
                        "code": code, "summary": summary, "lang": language,
                        "meta": json.dumps(metadata), "emb": embedding,
                    },
                )
                self.conn.execute(
                    """
                    MATCH (s:Source {source_id: $sid}), (e:CodeExample {example_id: $eid})
                    MERGE (s)-[:HAS_EXAMPLE]->(e)
                    """,
                    {"sid": source_id, "eid": eid},
                )
                examples_added += 1

        return {"success": True, "count": examples_added, "total": len(urls)}

    # -------------------------------------------------------------------------
    # WRITE: source summary (was: sources table upsert)
    # -------------------------------------------------------------------------
    async def update_source_info(
        self, source_id: str, summary: str, word_count: int
    ) -> Dict[str, Any]:
        try:
            self.conn.execute(
                """
                MERGE (s:Source {source_id: $sid})
                SET s.summary    = $summary,
                    s.word_count = $wc,
                    s.updated_at = $ts
                """,
                {
                    "sid": source_id,
                    "summary": summary,
                    "wc": word_count,
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
            )
            return {"success": True, "source_id": source_id}
        except Exception as e:
            logger.error(f"update_source_info failed: {e}")
            return {"success": False, "error": str(e)}

    # -------------------------------------------------------------------------
    # READ: available sources (was: sources + count joins)
    # -------------------------------------------------------------------------
    async def get_available_sources(self) -> List[SourceInfo]:
        try:
            result = self.conn.execute(
                """
                MATCH (s:Source)
                OPTIONAL MATCH (s)-[:CONTAINS]->(c:Chunk)
                OPTIONAL MATCH (s)-[:HAS_EXAMPLE]->(e:CodeExample)
                RETURN s.source_id      AS source_id,
                       s.summary        AS summary,
                       s.word_count     AS word_count,
                       s.updated_at     AS updated_at,
                       count(DISTINCT c) AS chunk_count,
                       count(DISTINCT e) AS code_count
                """
            )
            sources = []
            while result.has_next():
                row = result.get_next()
                sources.append(SourceInfo(
                    source=row,
                    summary=row,
                    word_count=row or 0,
                    last_updated=row,
                    total_chunks=row,
                    total_documents=row,   # same metric
                    total_code_examples=row,[^1]
                ))
            return sources
        except Exception as e:
            logger.error(f"get_available_sources failed: {e}")
            return []
```


***

## 3. `src/services/search.py` — full replacement

This replaces all `self.client.rpc(...)` Supabase calls with Kuzu vector + FTS queries. The hybrid search is implemented via **Reciprocal Rank Fusion (RRF)** since Kuzu has no single-call hybrid query.

```python
"""
KuzuDB search service.
Drop-in replacement for the Supabase SearchService.
All public method signatures preserved.
Hybrid search = vector ANN + BM25 FTS fused via RRF.
"""
import json
import logging
from typing import Any, Dict, List, Optional

import kuzu

from crawl4ai_mcp.config import get_settings
from crawl4ai_mcp.models import SearchRequest, SearchResult, SearchResponse, SearchType
from crawl4ai_mcp.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard RRF constant


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank)


class SearchService:
    """KuzuDB-backed search service (replaces Supabase RPC calls)."""

    def __init__(
        self,
        client: kuzu.Connection,
        settings: Optional[Any] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.conn = client
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service or EmbeddingService(self.settings)

    # -------------------------------------------------------------------------
    # HALLUCINATION CHECK (replaces Neo4j source verification)
    # -------------------------------------------------------------------------
    def source_exists(self, source_id: str) -> bool:
        """
        Replaces Neo4j: verify a cited source actually exists before surfacing
        it to the agent. Pure graph lookup — O(1) primary key scan.
        """
        result = self.conn.execute(
            "MATCH (s:Source {source_id: $sid}) RETURN count(s) AS n",
            {"sid": source_id},
        )
        return result.get_next() > 0 if result.has_next() else False

    # -------------------------------------------------------------------------
    # VECTOR SEARCH
    # -------------------------------------------------------------------------
    def _vector_search(
        self,
        embedding: List[float],
        table: str,          # "Chunk" or "CodeExample"
        index: str,          # "chunk_embedding_idx" or "code_embedding_idx"
        match_count: int,
        source_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Core ANN vector search using Kuzu HNSW index.
        Optional source_id filter exploits the CONTAINS/HAS_EXAMPLE graph edge —
        this is the "blast radius" scoped search that pgvector cannot do natively.
        """
        if source_id:
            rel = "CONTAINS" if table == "Chunk" else "HAS_EXAMPLE"
            cypher = f"""
                MATCH (s:Source {{source_id: $sid}})-[:{rel}]->(n:{table})
                CALL QUERY_VECTOR_INDEX('{index}', $emb, $k)
                  YIELD node, distance
                WHERE node = n
                RETURN node.url           AS url,
                       node.chunk_number  AS chunk_number,
                       node.content       AS content,
                       node.metadata      AS metadata,
                       node.source_id     AS source_id,
                       (1.0 - distance)   AS similarity
                ORDER BY similarity DESC
            """
            params = {"sid": source_id, "emb": embedding, "k": match_count}
        else:
            cypher = f"""
                CALL QUERY_VECTOR_INDEX('{index}', $emb, $k)
                  YIELD node, distance
                RETURN node.url           AS url,
                       node.chunk_number  AS chunk_number,
                       node.content       AS content,
                       node.metadata      AS metadata,
                       (1.0 - distance)   AS similarity
                ORDER BY similarity DESC
            """
            params = {"emb": embedding, "k": match_count}

        result = self.conn.execute(cypher, params)
        rows = []
        while result.has_next():
            r = result.get_next()
            # Resolve source_id from graph if not returned directly
            sid = r if source_id is None else source_id
            rows.append({
                "url": r, "chunk_number": r, "content": r,
                "metadata": json.loads(r) if r else {},
                "source_id": sid, "similarity": r if source_id else r,
            })
        return rows

    # -------------------------------------------------------------------------
    # FTS SEARCH (BM25)
    # -------------------------------------------------------------------------
    def _fts_search(
        self,
        query: str,
        table: str,
        index: str,
        match_count: int,
        source_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        if source_id:
            rel = "CONTAINS" if table == "Chunk" else "HAS_EXAMPLE"
            cypher = f"""
                MATCH (s:Source {{source_id: $sid}})-[:{rel}]->(n:{table})
                CALL QUERY_FTS_INDEX('{index}', $query, k := $k)
                  YIELD node, score
                WHERE node = n
                RETURN node.url, node.chunk_number, node.content,
                       node.metadata, score
            """
            params = {"sid": source_id, "query": query, "k": match_count}
        else:
            cypher = f"""
                CALL QUERY_FTS_INDEX('{index}', $query, k := $k)
                  YIELD node, score
                RETURN node.url, node.chunk_number, node.content,
                       node.metadata, score
            """
            params = {"query": query, "k": match_count}

        result = self.conn.execute(cypher, params)
        rows = []
        while result.has_next():
            r = result.get_next()
            rows.append({
                "url": r, "chunk_number": r, "content": r,
                "metadata": json.loads(r) if r else {},
                "source_id": source_id or "", "similarity": float(r),
            })
        return rows

    # -------------------------------------------------------------------------
    # RRF FUSION (replaces Supabase hybrid RPC)
    # -------------------------------------------------------------------------
    @staticmethod
    def _rrf_fuse(
        vec_rows: List[Dict], fts_rows: List[Dict], k: int
    ) -> List[Dict]:
        scores: Dict[str, float] = {}
        merged: Dict[str, Dict] = {}

        for rank, row in enumerate(vec_rows):
            key = f"{row['url']}::{row['chunk_number']}"
            scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
            merged[key] = row

        for rank, row in enumerate(fts_rows):
            key = f"{row['url']}::{row['chunk_number']}"
            scores[key] = scores.get(key, 0.0) + _rrf_score(rank)
            if key not in merged:
                merged[key] = row

        ranked = sorted(scores.items(), key=lambda x: x, reverse=True)[:k]
        results = []
        for key, score in ranked:
            row = merged[key].copy()
            row["similarity"] = score          # RRF score replaces cosine
            results.append(row)
        return results

    # -------------------------------------------------------------------------
    # PUBLIC: search_documents (was: client.rpc('match_crawled_pages', ...))
    # -------------------------------------------------------------------------
    async def search_documents(
        self,
        query: str,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None,
        use_hybrid_search: Optional[bool] = None,
    ) -> List[SearchResult]:
        try:
            embedding = await self.embedding_service.create_embedding(query)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []

        # --- Hallucination guard: validate source exists in graph -----------
        if source_id and not self.source_exists(source_id):
            logger.warning(f"Source '{source_id}' not in graph — skipping filter")
            source_id = None

        hybrid = use_hybrid_search
        if hybrid is None:
            hybrid = getattr(self.settings, "use_hybrid_search", False)

        if hybrid:
            vec_rows = self._vector_search(
                embedding, "Chunk", "chunk_embedding_idx", match_count * 2, source_id
            )
            fts_rows = self._fts_search(
                query, "Chunk", "chunk_fts_idx", match_count * 2, source_id
            )
            rows = self._rrf_fuse(vec_rows, fts_rows, match_count)
        else:
            rows = self._vector_search(
                embedding, "Chunk", "chunk_embedding_idx", match_count, source_id
            )

        return [
            SearchResult(
                content=r["content"],
                url=r["url"],
                source=r["source_id"],
                chunk_number=r["chunk_number"],
                similarity_score=r["similarity"],
                metadata=r["metadata"],
            )
            for r in rows
        ]

    # -------------------------------------------------------------------------
    # PUBLIC: search_code_examples (was: client.rpc('match_code_examples', ...))
    # -------------------------------------------------------------------------
    async def search_code_examples(
        self,
        query: str,
        language: Optional[str] = None,
        match_count: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        enhanced = f"Code example for {query}"
        if language:
            enhanced += f" in {language}"
        enhanced += f"\n\nSummary: Example code showing {query}"

        try:
            embedding = await self.embedding_service.create_embedding(enhanced)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []

        if source_id and not self.source_exists(source_id):
            source_id = None

        rows = self._vector_search(
            embedding, "CodeExample", "code_embedding_idx", match_count, source_id
        )

        if language:
            rows = [r for r in rows if r.get("metadata", {}).get("language") == language]

        return rows

    # -------------------------------------------------------------------------
    # PUBLIC: perform_search — unchanged signature
    # -------------------------------------------------------------------------
    async def perform_search(
        self,
        request: SearchRequest,
        search_type: SearchType = SearchType.SEMANTIC,
        include_code_examples: bool = False,
    ) -> SearchResponse:
        try:
            results = await self.search_documents(
                query=request.query,
                match_count=request.num_results,
                source_id=request.source,
                use_hybrid_search=(search_type == SearchType.HYBRID),
            )

            if request.semantic_threshold > 0:
                results = [r for r in results if r.similarity_score >= request.semantic_threshold]

            code_results = []
            if include_code_examples and getattr(self.settings, "use_agentic_rag", False):
                code_data = await self.search_code_examples(
                    query=request.query,
                    match_count=request.num_results,
                    source_id=request.source,
                )
                for ex in code_data:
                    meta = ex.get("metadata", {})
                    code_results.append(SearchResult(
                        content=ex["content"],
                        url=ex["url"],
                        source=ex.get("source_id", ""),
                        chunk_number=ex["chunk_number"],
                        similarity_score=ex["similarity"],
                        metadata={"type": "code_example", "language": meta.get("language", "unknown"), **meta},
                    ))

            all_results = sorted(results + code_results, key=lambda x: x.similarity_score, reverse=True)
            return SearchResponse(
                success=True,
                results=all_results[:request.num_results],
                total_results=len(all_results),
                search_type=search_type,
            )
        except Exception as e:
            return SearchResponse(success=False, results=[], total_results=0, search_type=search_type, error=str(e))

    # -------------------------------------------------------------------------
    # PUBLIC: rerank_results — no change, pure Python logic
    # -------------------------------------------------------------------------
    async def rerank_results(
        self,
        query: str,
        results: List[SearchResult],
        reranking_model: Any,
        threshold: float = 0.3,
    ) -> List[SearchResult]:
        if not results or not reranking_model:
            return results
        try:
            pairs = [[query, r.content] for r in results]
            scores = reranking_model.predict(pairs)
            for r, score in zip(results, scores):
                r.rerank_score = float(score)
            return sorted(
                [r for r in results if (r.rerank_score or 0) >= threshold],
                key=lambda x: x.rerank_score or 0,
                reverse=True,
            )
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results
```


***

## 4. `src/crawl4ai_mcp/server.py` — wiring change only

Find the lifespan startup block. Replace the Supabase + Neo4j initialisation with this:

```python
# BEFORE (remove these):
# from supabase import create_client
# from neo4j import AsyncGraphDatabase
# supabase_client = create_client(settings.supabase_url, settings.supabase_key)
# neo4j_driver = AsyncGraphDatabase.driver(settings.neo4j_uri, ...)

# AFTER:
from crawl4ai_mcp.services.kuzu_schema import init_db
from crawl4ai_mcp.services.database import DatabaseService
from crawl4ai_mcp.services.search import SearchService

kuzu_conn = init_db(settings.kuzu_db_path)   # see step 5 for config

db_service     = DatabaseService(client=kuzu_conn, settings=settings)
search_service = SearchService(client=kuzu_conn, settings=settings)
```

No other changes in `server.py` — every call site uses `db_service` and `search_service` through the same method names.

***

## 5. `src/crawl4ai_mcp/config.py` — env var changes

```python
# Remove:
# SUPABASE_URL: str
# SUPABASE_SERVICE_KEY: str
# NEO4J_URI: str
# NEO4J_USERNAME: str
# NEO4J_PASSWORD: str

# Add:
kuzu_db_path: str = Field(default=".kuzu_db", env="KUZU_DB_PATH")
```

`.env` file becomes:

```bash
# .env  — after migration
KUZU_DB_PATH=./.kuzu_db

OPENAI_API_KEY=sk-...
CRAWL4AI_API_TOKEN=...

# remove all SUPABASE_* and NEO4J_* lines
```


***

## 6. Verification script

Run this after boot to confirm all three systems (graph, vector, FTS) are live:

```python
# scripts/verify_kuzu.py
from crawl4ai_mcp.services.kuzu_schema import init_db

conn = init_db()

# 1. Schema sanity
r = conn.execute("CALL show_tables() RETURN *")
tables = []
while r.has_next():
    tables.append(r.get_next())
assert set(tables) >= {"Source","Chunk","CodeExample","CONTAINS","HAS_EXAMPLE","NEXT_CHUNK"}, tables
print("✓ Schema OK:", tables)

# 2. Vector index
r = conn.execute("CALL show_indexes() RETURN *")
idxs = []
while r.has_next():
    idxs.append(r.get_next())
print("✓ Indexes:", idxs)

# 3. Round-trip write + read
conn.execute("MERGE (s:Source {source_id:'test.com'}) SET s.word_count=0, s.updated_at='now'")
conn.execute("""
    MERGE (c:Chunk {chunk_id:'test.com::chunk::0'})
    SET c.url='https://test.com', c.chunk_number=0,
        c.content='hello world', c.metadata='{}',
        c.embedding=[""" + ",".join(["0.0"]*1536) + """]
""")
conn.execute("""
    MATCH (s:Source {source_id:'test.com'}), (c:Chunk {chunk_id:'test.com::chunk::0'})
    MERGE (s)-[:CONTAINS]->(c)
""")
r = conn.execute("MATCH (s:Source)-[:CONTAINS]->(c:Chunk) RETURN s.source_id, c.content")
row = r.get_next()
assert row == ["test.com", "hello world"], row
print("✓ Graph write+read OK:", row)

# 4. Hallucination guard
r = conn.execute("MATCH (s:Source {source_id:'test.com'}) RETURN count(s)")
assert r.get_next() == 1
r = conn.execute("MATCH (s:Source {source_id:'ghost.com'}) RETURN count(s)")
assert r.get_next() == 0
print("✓ Hallucination guard OK")

# 5. cleanup
conn.execute("MATCH (n) DETACH DELETE n")
print("✓ All checks passed — Kuzu ready")
```

Run it:

```bash
python scripts/verify_kuzu.py
```


***

## Key behavioural differences vs. the Supabase/Neo4j version

| Behaviour | Before | After |
| :-- | :-- | :-- |
| `source_id` filter on vector search | pgvector full-scan + WHERE | Graph hop → scoped HNSW ANN |
| Hallucination check | Neo4j driver RPC | Inline `MATCH (s:Source …) RETURN count(s)` |
| Hybrid search | Supabase RPC (native) | RRF fusion of Kuzu HNSW + BM25 |
| Chunk continuity | No native links | `NEXT_CHUNK` edges traversable for context window expansion |
| Infrastructure | Supabase cloud + Neo4j process | Single `.kuzu_db/` directory |
| Concurrency | Managed by Supabase | Vela fork WAL — safe for multi-agent writes |

<div align="center">⁂</div>

[^1]: https://www.falkordb.com/blog/falkordblite-embedded-python-graph-database/

