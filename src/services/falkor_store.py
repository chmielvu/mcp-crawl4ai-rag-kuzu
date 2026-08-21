"""FalkorDB graph store for crawl-graph ontology ingestion and management."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
import inspect
from typing import Any
from urllib.parse import urlparse, urlunparse

from falkordb.asyncio import FalkorDB
from redis.asyncio import BlockingConnectionPool

from crawl4ai_mcp.services.contracts import (
    CrawlIngestion,
    GraphOperationResult,
    SiteInfo,
    SitePayload,
)

logger = logging.getLogger(__name__)


def canonicalize_url(url: str) -> str:
    """Normalize a URL to its canonical form for deterministic graph keys."""
    if not url:
        return ""
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    if ":" in netloc:
        host, port = netloc.split(":", 1)
        if (scheme == "http" and port == "80") or (scheme == "https" and port == "443"):
            netloc = host
    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))


def deterministic_page_id(canonical_url: str) -> str:
    """Generate a deterministic page ID from canonical URL."""
    return hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:16]


def deterministic_content_hash(content: str) -> str:
    """Generate a deterministic SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def deterministic_chunk_id(
    page_id: str, index: int, content_type: str = "text"
) -> str:
    """Generate the same deterministic chunk ID used by ingestion payloads."""
    return f"{page_id}::{content_type}::{index}"

def deterministic_entity_id(name: str, entity_type: str) -> str:
    """Generate an entity ID from normalized name and type."""
    normalized = f"{name.strip().casefold()}\x00{entity_type.strip().casefold()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]



def _parse_timestamp(value: Any) -> datetime | None:
    """Parse string/datetime into UTC datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _parse_json_dict(value: Any) -> dict[str, Any]:
    """Safely parse JSON dictionary or return empty dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _extract_rows(result: Any) -> list[list[Any]]:
    """Extract tabular rows from FalkorDB QueryResult or mock object."""
    if result is None:
        return []
    if hasattr(result, "result_set") and isinstance(result.result_set, list):
        return result.result_set
    if isinstance(result, list):
        return result
    return []


def _extract_headers(result: Any) -> list[str]:
    """Extract column header names from FalkorDB QueryResult or mock object."""
    if result is None:
        return []
    if hasattr(result, "header") and result.header:
        headers: list[str] = []
        for col in result.header:
            if isinstance(col, (list, tuple)):
                values = list(col)
                if values:
                    headers.append(str(values[1] if len(values) > 1 else values[0]))
            else:
                headers.append(str(col))
        return headers
    if hasattr(result, "columns") and result.columns:
        return [str(c) for c in result.columns]
    return []


def _query_result_to_dicts(result: Any) -> list[dict[str, Any]]:
    """Convert FalkorDB QueryResult rows to list of dictionaries with normalized keys."""
    rows = _extract_rows(result)
    headers = _extract_headers(result)
    if not headers or not rows:
        return []
    dicts: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            norm_dict: dict[str, Any] = {}
            for k, v in row.items():
                norm_dict[k] = v
                if "." in k:
                    norm_dict[k.split(".", 1)[-1]] = v
            dicts.append(norm_dict)
        elif isinstance(row, (list, tuple)):
            row_dict: dict[str, Any] = {}
            for i in range(min(len(headers), len(row))):
                header = headers[i]
                val = row[i]
                row_dict[header] = val
                if "." in header:
                    row_dict[header.split(".", 1)[-1]] = val
            dicts.append(row_dict)
    return dicts


class FalkorStore:
    """FalkorDB graph store implementing GraphStorePort."""

    def __init__(
        self,
        graph: Any = None,
        db: FalkorDB | Any = None,
        connection_pool: BlockingConnectionPool | Any = None,
        host: str = "localhost",
        port: int = 6379,
        password: str | None = None,
        graph_name: str = "crawl-graph",
        max_connections: int = 16,
        decode_responses: bool = True,
        settings: Any = None,
        **pool_kwargs: Any,
    ):
        self._graph = graph
        self._db = db
        self._pool = connection_pool
        self.settings = settings
        self.graph_name = graph_name

        if self._graph is None:
            if self._db is None:
                if self._pool is None:
                    self._pool = BlockingConnectionPool(
                        host=host,
                        port=port,
                        password=password,
                        max_connections=max_connections,
                        decode_responses=decode_responses,
                        **pool_kwargs,
                    )
                self._db = FalkorDB(connection_pool=self._pool)
            self._graph = self._db.select_graph(graph_name)

    @property
    def graph(self) -> Any:
        """Return the active graph instance."""
        return self._graph

    async def _query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a write/update Cypher query on the graph."""
        query_method = getattr(self._graph, "query", None)
        if callable(query_method):
            timeout = getattr(self.settings, "falkordb_query_timeout_ms", None)
            return await query_method(query, params=params, timeout=timeout)
        execute_method = getattr(self._graph, "execute", None)
        if callable(execute_method):
            return await execute_method(query, params=params)
        raise AttributeError(
            f"Graph object {type(self._graph)} has neither query nor execute method"
        )

    async def _ro_query(self, query: str, params: dict[str, Any] | None = None) -> Any:
        """Execute a read-only Cypher query on the graph."""
        timeout = getattr(self.settings, "falkordb_query_timeout_ms", None)
        ro_query_method = getattr(self._graph, "ro_query", None)
        if callable(ro_query_method) and inspect.iscoroutinefunction(ro_query_method):
            return await ro_query_method(query, params=params, timeout=timeout)
        query_method = getattr(self._graph, "query", None)
        if callable(query_method):
            return await query_method(query, params=params, timeout=timeout)
        execute_method = getattr(self._graph, "execute", None)
        if callable(execute_method):
            return await execute_method(query, params=params)
        raise AttributeError(f"Graph object {type(self._graph)} has no query method")

    async def ingest_crawl(self, payload: CrawlIngestion) -> GraphOperationResult:
        """Ingest atomic site crawl with GLiNER facts, page/chunk/link provenance, MENTIONED_IN and RELATES."""
        try:
            site = payload.site
            total_chunks = sum(len(p.chunks) for p in payload.pages)
            total_pages = len(payload.pages)

            # 1. Upsert Site node
            await self._upsert_site(site, total_pages, total_chunks, payload.finished_at)

            # 2. Upsert CrawlRun node and (:Site)-[:HAS_RUN]->(:CrawlRun)
            await self._upsert_crawl_run(payload)

            # 3. UNWIND batch upsert Pages and relationships
            pages_data = await self._upsert_pages(payload)

            # 4. UNWIND batch upsert Chunks and (:Page)-[:HAS_CHUNK]->(:Chunk)
            chunks_data = await self._upsert_chunks(payload)


            # 6. UNWIND batch upsert page links (:Page)-[:LINKS_TO]->(:Page)
            links_data = await self._upsert_links(payload)

            # 7. Grounded MENTIONED_IN: (:__Entity__)-[:MENTIONED_IN]->(:Chunk)
            extractions_count = await self._upsert_chunk_extractions(payload)

            # 8. Site-level GLiNER entities and RELATES facts
            entities_count, relations_count = await self._upsert_site_gliner(payload)

            total_entities = extractions_count + entities_count

            return GraphOperationResult(
                success=True,
                run_id=payload.run_id,
                pages=len(pages_data),
                chunks=len(chunks_data),
                entities=total_entities,
                relations=relations_count,
                links=len(links_data),
                details={
                    "site_id": site.site_id,
                    "root_url": payload.root_url,
                    "pages_stored": len(pages_data),
                    "chunks_stored": len(chunks_data),
                },
            )
        except Exception as exc:
            logger.error("Failed to ingest crawl payload for run %s: %s", payload.run_id, exc, exc_info=True)
            return GraphOperationResult(
                success=False,
                run_id=payload.run_id,
                error=str(exc),
                details={"root_url": payload.root_url},
            )

    async def _upsert_site(
        self,
        site: SitePayload,
        page_count: int,
        chunk_count: int,
        finished_at: datetime,
    ) -> None:
        """Upsert Site node with metadata and counts."""
        query = (
            "MERGE (s:Site {site_id: $site_id}) "
            "ON CREATE SET "
            "  s.domain = $domain, "
            "  s.root_url = $root_url, "
            "  s.title = $title, "
            "  s.summary = $summary, "
            "  s.first_seen = $first_seen, "
            "  s.last_crawled = $last_crawled, "
            "  s.gliner_metadata = $gliner_metadata, "
            "  s.page_count = $page_count, "
            "  s.chunk_count = $chunk_count "
            "ON MATCH SET "
            "  s.domain = $domain, "
            "  s.root_url = $root_url, "
            "  s.title = coalesce($title, s.title), "
            "  s.summary = coalesce($summary, s.summary), "
            "  s.last_crawled = $last_crawled, "
            "  s.gliner_metadata = $gliner_metadata, "
            "  s.page_count = $page_count, "
            "  s.chunk_count = $chunk_count"
        )
        params = {
            "site_id": site.site_id,
            "domain": site.domain,
            "root_url": site.root_url,
            "title": site.title,
            "summary": site.summary,
            "first_seen": finished_at.isoformat(),
            "last_crawled": finished_at.isoformat(),
            "gliner_metadata": json.dumps(site.gliner_metadata) if site.gliner_metadata else "{}",
            "page_count": page_count,
            "chunk_count": chunk_count,
        }
        await self._query(query, params=params)

    async def _upsert_crawl_run(self, payload: CrawlIngestion) -> None:
        """Upsert a CrawlRun node before page provenance writes."""
        query = (
            "MERGE (r:CrawlRun {run_id: $run_id}) "
            "ON CREATE SET "
            "  r.root_url = $root_url, "
            "  r.started_at = $started_at, "
            "  r.finished_at = $finished_at, "
            "  r.status = 'completed', "
            "  r.max_depth = $max_depth, "
            "  r.pages_crawled = $pages_crawled "
            "ON MATCH SET "
            "  r.finished_at = $finished_at, "
            "  r.status = 'completed', "
            "  r.pages_crawled = $pages_crawled"
        )
        await self._query(
            query,
            params={
                "run_id": payload.run_id,
                "root_url": payload.root_url,
                "started_at": payload.started_at.isoformat(),
                "finished_at": payload.finished_at.isoformat(),
                "max_depth": payload.max_depth,
                "pages_crawled": len(payload.pages),
            },
        )

    async def _upsert_pages(self, payload: CrawlIngestion) -> list[dict[str, Any]]:
        """UNWIND batch upsert Page nodes and link to Site and CrawlRun."""
        pages_data: list[dict[str, Any]] = []
        for p in payload.pages:
            canonical = p.canonical_url or canonicalize_url(p.url)
            page_id = p.page_id or deterministic_page_id(canonical)
            content_hash = p.content_hash or deterministic_content_hash(p.title or canonical)
            pages_data.append({
                "page_id": page_id,
                "url": p.url,
                "canonical_url": canonical,
                "title": p.title or "",
                "status_code": p.status_code or 200,
                "content_type": p.content_type or "text/html",
                "language": p.language or "en",
                "content_hash": content_hash,
                "depth": p.depth,
                "crawled_at": p.crawled_at.isoformat() if p.crawled_at else payload.finished_at.isoformat(),
                "metadata_json": p.metadata_json if p.metadata_json else "{}",
            })

        if not pages_data:
            return []

        query = (
            "UNWIND $pages AS p_data "
            "MERGE (p:Page {page_id: p_data.page_id}) "
            "ON CREATE SET "
            "  p.url = p_data.url, "
            "  p.canonical_url = p_data.canonical_url, "
            "  p.title = p_data.title, "
            "  p.status_code = p_data.status_code, "
            "  p.content_type = p_data.content_type, "
            "  p.language = p_data.language, "
            "  p.content_hash = p_data.content_hash, "
            "  p.depth = p_data.depth, "
            "  p.crawled_at = p_data.crawled_at, "
            "  p.metadata_json = p_data.metadata_json "
            "ON MATCH SET "
            "  p.url = p_data.url, "
            "  p.canonical_url = p_data.canonical_url, "
            "  p.title = coalesce(p_data.title, p.title), "
            "  p.status_code = p_data.status_code, "
            "  p.content_type = p_data.content_type, "
            "  p.language = p_data.language, "
            "  p.content_hash = p_data.content_hash, "
            "  p.depth = p_data.depth, "
            "  p.crawled_at = p_data.crawled_at, "
            "  p.metadata_json = p_data.metadata_json "
            "WITH p, p_data "
            "MATCH (s:Site {site_id: $site_id}) "
            "MERGE (s)-[:HAS_PAGE]->(p) "
            "WITH p, p_data "
            "MATCH (r:CrawlRun {run_id: $run_id}) "
            "MERGE (r)-[:CRAWLED]->(p)"
        )
        await self._query(query, params={"pages": pages_data, "site_id": payload.site.site_id, "run_id": payload.run_id})
        return pages_data

    async def _upsert_chunks(self, payload: CrawlIngestion) -> list[dict[str, Any]]:
        """UNWIND batch upsert Chunk nodes with vecf32 vector writes and link to Page."""
        chunks_data: list[dict[str, Any]] = []
        for p in payload.pages:
            canonical = p.canonical_url or canonicalize_url(p.url)
            page_id = p.page_id or deterministic_page_id(canonical)
            for c in p.chunks:
                chunk_id = c.chunk_id or deterministic_chunk_id(
                    page_id, c.index, c.content_type
                )
                chunks_data.append({
                    "chunk_id": chunk_id,
                    "page_id": page_id,
                    "text": c.text,
                    "index": c.index,
                    "heading_path": c.heading_path or "",
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "content_type": c.content_type or "text",
                    "language": c.language or "en",
                    "metadata_json": c.metadata_json if c.metadata_json else "{}",
                    "embedding": c.embedding,
                })

        if not chunks_data:
            return []

        query = (
            "UNWIND $chunks AS c_data "
            "MERGE (c:Chunk {chunk_id: c_data.chunk_id}) "
            "ON CREATE SET "
            "  c.text = c_data.text, "
            "  c.index = c_data.index, "
            "  c.heading_path = c_data.heading_path, "
            "  c.start_char = c_data.start_char, "
            "  c.end_char = c_data.end_char, "
            "  c.content_type = c_data.content_type, "
            "  c.language = c_data.language, "
            "  c.metadata_json = c_data.metadata_json, "
            "  c.embedding = vecf32(c_data.embedding) "
            "ON MATCH SET "
            "  c.text = c_data.text, "
            "  c.index = c_data.index, "
            "  c.heading_path = c_data.heading_path, "
            "  c.start_char = c_data.start_char, "
            "  c.end_char = c_data.end_char, "
            "  c.content_type = c_data.content_type, "
            "  c.language = c_data.language, "
            "  c.metadata_json = c_data.metadata_json, "
            "  c.embedding = vecf32(c_data.embedding) "
            "WITH c, c_data "
            "MATCH (p:Page {page_id: c_data.page_id}) "
            "MERGE (p)-[:HAS_CHUNK]->(c)"
        )
        await self._query(query, params={"chunks": chunks_data})
        return chunks_data


    async def _upsert_links(self, payload: CrawlIngestion) -> list[dict[str, Any]]:
        """UNWIND batch upsert page links (:Page)-[:LINKS_TO]->(:Page) for graph provenance."""
        links_data: list[dict[str, Any]] = []
        for p in payload.pages:
            canonical_src = p.canonical_url or canonicalize_url(p.url)
            src_page_id = p.page_id or deterministic_page_id(canonical_src)
            for link in p.links:
                if not link.href:
                    continue
                target_canonical = canonicalize_url(link.href)
                target_page_id = deterministic_page_id(target_canonical)
                links_data.append(
                    {
                        "source_page_id": src_page_id,
                        "target_page_id": target_page_id,
                        "href": link.href,
                        "target_canonical": target_canonical,
                        "anchor_text": link.text or link.title or "",
                        "rel": link.rel or "",
                        "discovered_at": p.crawled_at.isoformat(),
                    }
                )

        if not links_data:
            return []

        query = (
            "UNWIND $links AS l_data "
            "MATCH (src:Page {page_id: l_data.source_page_id}) "
            "MERGE (dst:Page {page_id: l_data.target_page_id}) "
            "ON CREATE SET "
            "  dst.url = l_data.href, "
            "  dst.canonical_url = l_data.target_canonical "
            "MERGE (src)-[r:LINKS_TO {anchor_text: l_data.anchor_text}]->(dst) "
            "SET r.rel = l_data.rel, "
            "    r.discovered_at = l_data.discovered_at"
        )
        await self._query(query, params={"links": links_data})
        return links_data

    async def _upsert_chunk_extractions(self, payload: CrawlIngestion) -> int:
        """Write grounded LangExtract entities and MENTIONED_IN provenance."""
        extractions_data: list[dict[str, Any]] = []
        for page in payload.pages:
            page_id = page.page_id or deterministic_page_id(
                page.canonical_url or canonicalize_url(page.url)
            )
            for chunk in page.chunks:
                chunk_id = chunk.chunk_id or deterministic_chunk_id(page_id, chunk.index)
                for extraction in chunk.extractions:
                    name = extraction.extraction_text.strip()
                    if not name:
                        continue
                    entity_type = extraction.extraction_class.strip() or "entity"
                    extractions_data.append(
                        {
                            "entity_id": deterministic_entity_id(name, entity_type),
                            "name": name,
                            "entity_type": entity_type,
                            "chunk_id": chunk_id,
                            "extraction_class": entity_type,
                            "extraction_text": name,
                            "start_char": extraction.start_char,
                            "end_char": extraction.end_char,
                            "attributes_json": json.dumps(extraction.attributes),
                            "description": extraction.description or "",
                        }
                    )
        if not extractions_data:
            return 0

        query = (
            "UNWIND $extractions AS ext "
            "MERGE (e:__Entity__ {entity_id: ext.entity_id}) "
            "SET e.name = ext.name, "
            "    e.entity_type = ext.entity_type, "
            "    e.description = ext.description "
            "WITH e, ext "
            "MATCH (c:Chunk {chunk_id: ext.chunk_id}) "
            "MERGE (e)-[m:MENTIONED_IN {chunk_id: ext.chunk_id}]->(c) "
            "SET m.extraction_class = ext.extraction_class, "
            "    m.extraction_text = ext.extraction_text, "
            "    m.start_char = ext.start_char, "
            "    m.end_char = ext.end_char, "
            "    m.confidence = 1.0, "
            "    m.extraction_source = 'langextract', "
            "    m.attributes_json = ext.attributes_json, "
            "    m.description = ext.description"
        )
        await self._query(query, params={"extractions": extractions_data})
        return len(extractions_data)

    async def _upsert_site_gliner(self, payload: CrawlIngestion) -> tuple[int, int]:
        """Write site-level GLiNER entities and normalized RELATES facts."""
        entities_data: list[dict[str, Any]] = []
        for entity in payload.site.entities:
            name = entity.text.strip()
            entity_type = entity.label.strip() or "entity"
            if name:
                if entity.embedding is None or len(entity.embedding) != 384:
                    raise ValueError(
                        f"GLiNER entity {name!r} is missing a 384-dimensional embedding"
                    )
                entities_data.append(
                    {
                        "entity_id": deterministic_entity_id(name, entity_type),
                        "name": name,
                        "entity_type": entity_type,
                        "confidence": entity.score if entity.score is not None else 1.0,
                        "embedding": entity.embedding,
                    }
                )
        if entities_data:
            entity_query = (
                "UNWIND $site_entities AS item "
                "MERGE (e:__Entity__ {entity_id: item.entity_id}) "
                "SET e.name = item.name, "
                "    e.entity_type = item.entity_type, "
                "    e.description = coalesce(e.description, ''), "
                "    e.confidence = item.confidence, "
                "    e.embedding = vecf32(item.embedding) "
                "WITH e "
                "MATCH (s:Site {site_id: $site_id}) "
                "MERGE (s)-[:HAS_ENTITY]->(e)"
            )
            await self._query(
                entity_query,
                params={"site_entities": entities_data, "site_id": payload.site.site_id},
            )

        relations_data: list[dict[str, Any]] = []
        for relation in payload.site.relations:
            source = relation.source.strip()
            target = relation.target.strip()
            relation_type = relation.relation.strip()
            source_type = relation.source_entity_type or "entity"
            target_type = relation.target_entity_type or "entity"
            if source and target and relation_type:
                if relation.embedding is None or len(relation.embedding) != 384:
                    raise ValueError(
                        f"GLiNER relation {relation_type!r} is missing a 384-dimensional embedding"
                    )
                relations_data.append(
                    {
                        "source_id": deterministic_entity_id(source, source_type),
                        "target_id": deterministic_entity_id(target, target_type),
                        "source": source,
                        "target": target,
                        "source_type": source_type,
                        "target_type": target_type,
                        "rel_type": relation_type,
                        "confidence": relation.score if relation.score is not None else 1.0,
                        "fact": relation.fact or "",
                        "description": relation.description or "",
                        "embedding": relation.embedding,
                    }
                )
        if relations_data:
            relation_query = (
                "UNWIND $site_relations AS item "
                "MERGE (src:__Entity__ {entity_id: item.source_id}) "
                "SET src.name = item.source, src.entity_type = item.source_type "
                "MERGE (dst:__Entity__ {entity_id: item.target_id}) "
                "SET dst.name = item.target, dst.entity_type = item.target_type "
                "MERGE (src)-[r:RELATES {rel_type: item.rel_type}]->(dst) "
                "SET r.confidence = item.confidence, "
                "    r.fact = item.fact, "
                "    r.description = item.description, "
                "    r.extraction_source = 'gliner', "
                "    r.source_chunk_ids = '[]', "
                "    r.spans_json = '[]', "
                "    r.embedding = vecf32(item.embedding)"
            )
            await self._query(
                relation_query, params={"site_relations": relations_data}
            )
        return len(entities_data), len(relations_data)

    async def get_available_sites(self) -> list[SiteInfo]:
        """List all indexed sites with aggregated page/chunk counts and GLiNER metadata."""
        query = (
            "MATCH (s:Site) "
            "OPTIONAL MATCH (s)-[:HAS_PAGE]->(p:Page) "
            "OPTIONAL MATCH (p)-[:HAS_CHUNK]->(c:Chunk) "
            "RETURN "
            "  s.site_id AS site_id, "
            "  s.domain AS domain, "
            "  s.root_url AS root_url, "
            "  s.title AS title, "
            "  s.summary AS summary, "
            "  s.first_seen AS first_seen, "
            "  s.last_crawled AS last_crawled, "
            "  count(DISTINCT p) AS page_count, "
            "  count(DISTINCT c) AS chunk_count, "
            "  s.gliner_metadata AS gliner_metadata "
            "ORDER BY s.site_id"
        )
        result = await self._ro_query(query)
        rows = _query_result_to_dicts(result)
        sites: list[SiteInfo] = []
        for row in rows:
            site_id = str(row.get("site_id") or "")
            if not site_id:
                continue
            sites.append(
                SiteInfo(
                    site_id=site_id,
                    domain=str(row.get("domain") or site_id),
                    root_url=str(row.get("root_url") or ""),
                    title=row.get("title") or None,
                    summary=row.get("summary") or None,
                    first_seen=_parse_timestamp(row.get("first_seen")),
                    last_crawled=_parse_timestamp(row.get("last_crawled")),
                    page_count=int(row.get("page_count") or 0),
                    chunk_count=int(row.get("chunk_count") or 0),
                    gliner_metadata=_parse_json_dict(row.get("gliner_metadata")),
                )
            )
        return sites

    async def aclose(self) -> None:
        """Close FalkorDB client and Redis connection pool."""
        try:
            if self._db is not None and hasattr(self._db, "aclose"):
                await self._db.aclose()
        except Exception as exc:
            logger.warning("Error closing FalkorDB client: %s", exc)

        try:
            if self._pool is not None:
                if hasattr(self._pool, "aclose"):
                    await self._pool.aclose()
                elif hasattr(self._pool, "disconnect"):
                    await self._pool.disconnect()
        except Exception as exc:
            logger.warning("Error closing connection pool: %s", exc)
