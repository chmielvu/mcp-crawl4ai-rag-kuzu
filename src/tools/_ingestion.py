"""Shared typed ingestion helpers for crawl and search tools."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
from urllib.parse import urlparse
from crawl4ai_mcp.services.contracts import (
    ChatGeneratorPort,
    ChunkPayload,
    CrawlDocument,
    CrawlIngestion,
    EmbeddingPort,
    GlinerEntity,
    GlinerPort,
    GlinerRelation,
    GraphOperationResult,
    GraphStorePort,
    PageExtraction,
    PagePayload,
    RemoteLink,
    SitePayload,
)
from crawl4ai_mcp.services.crawling import CrawlingService
from crawl4ai_mcp.services.falkor_store import canonicalize_url
from crawl4ai_mcp.utilities.text_processing import TextProcessor

logger = logging.getLogger(__name__)

EXPECTED_EMBEDDING_DIMENSIONS = 384
MAX_EMBEDDING_BATCH_SIZE = 32


def deterministic_site_id(url_or_domain: str) -> str:
    """Derive deterministic site identifier from URL or domain."""
    netloc = urlparse(url_or_domain).netloc if "://" in url_or_domain else url_or_domain
    return netloc.strip().lower() or url_or_domain.strip().lower()


def deterministic_page_id(url: str) -> str:
    """Derive deterministic page identifier from canonical URL."""
    canonical_url = canonicalize_url(url)
    return hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:16]


def deterministic_chunk_id(page_id: str, index: int, content_type: str = "text") -> str:
    """Derive deterministic chunk identifier from page_id, content_type, and index."""
    return f"{page_id}::{content_type}::{index}"


def compute_content_hash(text: str) -> str:
    """Compute sha256 hash of document text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def embed_texts_in_batches(
    texts: Sequence[str],
    embedding_port: EmbeddingPort,
    batch_size: int = MAX_EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    """Embed texts in batches of <= 32 through EmbeddingPort with 384-dim check."""
    if not texts:
        return []

    effective_batch_size = max(1, min(MAX_EMBEDDING_BATCH_SIZE, batch_size))
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), effective_batch_size):
        batch = list(texts[start : start + effective_batch_size])
        batch_embeddings = await embedding_port.embed_passages(batch)
        for vector in batch_embeddings:
            if len(vector) != EXPECTED_EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {EXPECTED_EMBEDDING_DIMENSIONS}, "
                    f"got {len(vector)}"
                )
            all_embeddings.append(vector)

    return all_embeddings


async def _run_langextract_for_page(
    extractor: Any,
    text: str,
    settings: Any = None,
) -> list[PageExtraction]:
    """Execute LangExtract once per page; never swallow extraction failures."""
    if extractor is None:
        return []

    if callable(extractor):
        res = extractor(text)
        if asyncio_iscoroutine(res):
            raw_extractions = await res
        else:
            raw_extractions = res
    elif hasattr(extractor, "extract"):
        res = extractor.extract(text)
        if asyncio_iscoroutine(res):
            raw_extractions = await res
        else:
            raw_extractions = res
    else:
        return []

    extractions: list[PageExtraction] = []
    if isinstance(raw_extractions, list):
        for item in raw_extractions:
            if isinstance(item, PageExtraction):
                extractions.append(item)
            elif isinstance(item, dict):
                extractions.append(
                    PageExtraction(
                        extraction_class=str(
                            item.get("extraction_class")
                            or item.get("class")
                            or item.get("label")
                            or "entity"
                        ),
                        extraction_text=str(
                            item.get("extraction_text")
                            or item.get("text")
                            or ""
                        ),
                        start_char=int(item.get("start_char", 0)),
                        end_char=int(item.get("end_char", 0)),
                        attributes=item.get("attributes", {}),
                        description=item.get("description"),
                    )
                )
            elif hasattr(item, "extraction_class") and hasattr(item, "start_char"):
                extractions.append(
                    PageExtraction(
                        extraction_class=str(item.extraction_class),
                        extraction_text=str(getattr(item, "extraction_text", "")),
                        start_char=int(item.start_char),
                        end_char=int(getattr(item, "end_char", 0)),
                        attributes=getattr(item, "attributes", {}),
                        description=getattr(item, "description", None),
                    )
                )
    return extractions


def asyncio_iscoroutine(obj: Any) -> bool:
    """Check if object is a coroutine or awaitable."""
    import inspect
    return inspect.iscoroutine(obj) or inspect.isawaitable(obj)


def map_extractions_to_chunks(
    page_extractions: list[PageExtraction],
    chunks: list[ChunkPayload],
) -> None:
    """Map page-grounded extractions to intersecting chunks with relative offsets."""
    if not page_extractions or not chunks:
        return

    for ext in page_extractions:
        for chunk in chunks:
            # Check intersection between page span [ext.start_char, ext.end_char] and chunk [chunk.start_char, chunk.end_char]
            if ext.start_char < chunk.end_char and ext.end_char > chunk.start_char:
                rel_start = max(0, ext.start_char - chunk.start_char)
                rel_end = min(len(chunk.text), ext.end_char - chunk.start_char)
                if rel_end > rel_start:
                    chunk.extractions.append(
                        PageExtraction(
                            extraction_class=ext.extraction_class,
                            extraction_text=ext.extraction_text,
                            start_char=rel_start,
                            end_char=rel_end,
                            attributes=ext.attributes.copy(),
                            description=ext.description,
                        )
                    )


async def build_page_payload(
    url: str,
    markdown_content: str,
    embedding_port: EmbeddingPort,
    text_processor: TextProcessor,
    crawling_service: CrawlingService | None = None,
    settings: Any = None,
    depth: int = 0,
    title: str | None = None,
    status_code: int | None = None,
    content_type: str | None = None,
    language: str | None = None,
    links: list[RemoteLink] | None = None,
    lang_extract: Any = None,
    chunk_size: int = 5000,
) -> PagePayload:
    """Construct a fully typed PagePayload with embedded text and code chunks."""
    canonical_url = canonicalize_url(url)
    page_id = deterministic_page_id(canonical_url)
    content_hash = compute_content_hash(markdown_content)

    # 1. Generate text chunks with exact character offsets
    chunks_meta = text_processor.smart_chunk_with_offsets(
        markdown_content, chunk_size=chunk_size
    )

    text_inputs: list[str] = []
    chunk_skeletons: list[dict[str, Any]] = []

    use_contextual = getattr(settings, "use_contextual_embeddings", False)
    for meta in chunks_meta:
        chunk_text = meta["text"]
        if use_contextual:
            contextual_text, _ = await text_processor.generate_contextual_embedding(
                markdown_content, chunk_text
            )
            text_inputs.append(contextual_text)
        else:
            text_inputs.append(chunk_text)

        chunk_skeletons.append(meta)

    # 2. Extract code blocks if agentic RAG / code extraction is enabled
    use_agentic = getattr(settings, "use_agentic_rag", True)
    code_skeletons: list[dict[str, Any]] = []
    if use_agentic and crawling_service is not None:
        code_blocks = crawling_service.extract_code_blocks(markdown_content)
        for code_block in code_blocks:
            code_str = code_block["code"]
            lang = code_block["language"] or "unknown"
            formatted_code = (
                f"```{lang}\n{code_str}\n```" if lang else f"```\n{code_str}\n```"
            )
            summary = await crawling_service.generate_code_example_summary(
                code_str,
                code_block["context_before"],
                code_block["context_after"],
            )
            embed_text = f"{summary}\n\n{formatted_code}" if summary else formatted_code
            text_inputs.append(embed_text)
            code_skeletons.append(
                {
                    "code_text": formatted_code,
                    "language": lang,
                    "summary": summary,
                    "start_char": code_block.get("start_char", 0),
                    "end_char": code_block.get("end_char", 0),
                }
            )

    # 3. Batch embed all text and code chunks (<= 32 per batch, 384 dimensions)
    batch_size = getattr(settings, "unified_ml_batch_size", MAX_EMBEDDING_BATCH_SIZE)
    embeddings = await embed_texts_in_batches(
        text_inputs,
        embedding_port=embedding_port,
        batch_size=batch_size,
    )

    # 4. Assemble ChunkPayload objects for text chunks
    all_chunks: list[ChunkPayload] = []
    num_text_chunks = len(chunk_skeletons)

    for i, meta in enumerate(chunk_skeletons):
        chunk_id = deterministic_chunk_id(page_id, i, content_type="text")
        chunk_payload = ChunkPayload(
            chunk_id=chunk_id,
            text=meta["text"],
            index=meta["index"],
            heading_path=meta.get("heading_path", ""),
            start_char=meta.get("start_char", 0),
            end_char=meta.get("end_char", 0),
            content_type="text",
            language=language,
            metadata_json=json.dumps(
                {
                    "url": url,
                    "page_id": page_id,
                    "heading_path": meta.get("heading_path", ""),
                    "section_info": meta.get("section_info", {}),
                }
            ),
            embedding=embeddings[i],
            extractions=[],
        )
        all_chunks.append(chunk_payload)

    # 5. Assemble ChunkPayload objects for code chunks
    for j, code_meta in enumerate(code_skeletons):
        emb_index = num_text_chunks + j
        code_chunk_id = deterministic_chunk_id(page_id, j, content_type="code")
        code_payload = ChunkPayload(
            chunk_id=code_chunk_id,
            text=code_meta["code_text"],
            index=num_text_chunks + j,
            heading_path="",
            start_char=code_meta.get("start_char", 0),
            end_char=code_meta.get("end_char", 0),
            content_type="code",
            language=code_meta.get("language"),
            metadata_json=json.dumps(
                {
                    "url": url,
                    "page_id": page_id,
                    "summary": code_meta.get("summary", ""),
                    "language": code_meta.get("language"),
                }
            ),
            embedding=embeddings[emb_index],
            extractions=[],
        )
        all_chunks.append(code_payload)

    # Execute LangExtract once per enabled page and propagate grounded spans.
    if getattr(settings, "use_langextract_metadata", False):
        if lang_extract is None:
            raise RuntimeError(
                "LangExtract metadata is enabled but no LangExtract provider was initialized"
            )
        page_extractions = await _run_langextract_for_page(
            lang_extract, markdown_content, settings=settings
        )
        map_extractions_to_chunks(page_extractions, all_chunks)

    return PagePayload(
        page_id=page_id,
        url=url,
        canonical_url=canonical_url,
        title=title,
        status_code=status_code,
        content_type=content_type,
        language=language,
        content_hash=content_hash,
        depth=depth,
        crawled_at=datetime.now(timezone.utc),
        metadata_json=json.dumps({"url": url, "content_hash": content_hash}),
        chunks=all_chunks,
        links=links or [],
    )


async def build_crawl_ingestion(
    root_url: str,
    pages: Sequence[PagePayload],
    max_depth: int = 3,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    gliner_port: GlinerPort | None = None,
    chat_generator: ChatGeneratorPort | None = None,
    embedding_port: EmbeddingPort | None = None,
    settings: Any = None,
) -> CrawlIngestion:
    """Build an atomic graph payload and complete site-level enrichment first."""
    domain = urlparse(root_url).netloc or root_url
    site_id = deterministic_site_id(root_url)
    now = datetime.now(timezone.utc)
    crawl_start = started_at or now
    crawl_finish = finished_at or now
    # Site-level summaries and GLiNER should receive prose, not code chunks.
    page_text = "\n\n".join(
        chunk.text
        for page in pages
        for chunk in page.chunks
        if chunk.text and chunk.content_type != "code"
    )
    capped_text = page_text[:10000]

    site_summary: str | None = None
    if chat_generator is not None and capped_text:
        try:
            site_summary = await chat_generator.extract_source_summary(
                site_id, capped_text, max_length=500
            )
        except Exception as error:
            logger.warning("Site summary generation failed for %s: %s", site_id, error)

    gliner_enabled = getattr(settings, "use_gliner_metadata", True)
    gliner_entities: list[GlinerEntity] = []
    gliner_relations: list[GlinerRelation] = []
    gliner_metadata: dict[str, Any] = {
        "enabled": bool(gliner_enabled),
        "entities": [],
        "relation_extraction": [],
    }
    if gliner_enabled:
        if gliner_port is None:
            raise RuntimeError(
                "GLiNER metadata is enabled but no GlinerPort was initialized"
            )
        if capped_text:
            entity_labels = getattr(
                settings,
                "gliner_entities",
                ("product", "technology", "library", "organization", "person"),
            )
            relation_labels = getattr(
                settings,
                "gliner_relations",
                ("uses", "depends_on", "implements", "stores"),
            )
            extraction = await gliner_port.extract(
                capped_text,
                entities=tuple(entity_labels),
                relations=tuple(relation_labels),
                threshold=getattr(settings, "gliner_threshold", 0.5),
                include_confidence=getattr(settings, "gliner_include_confidence", True),
                include_spans=getattr(settings, "gliner_include_spans", True),
            )
            gliner_entities = extraction.entities
            gliner_relations = extraction.relation_extraction
            gliner_metadata = extraction.model_dump(mode="json")

    known_entities = {
        (entity.text.casefold(), entity.label.casefold())
        for entity in gliner_entities
    }
    for relation in gliner_relations:
        for name, entity_type in (
            (relation.source, relation.source_entity_type or "entity"),
            (relation.target, relation.target_entity_type or "entity"),
        ):
            key = (name.casefold(), entity_type.casefold())
            if name and key not in known_entities:
                gliner_entities.append(
                    GlinerEntity(text=name, label=entity_type)
                )
                known_entities.add(key)

    if gliner_entities or gliner_relations:
        if embedding_port is None:
            raise RuntimeError(
                "GLiNER facts were extracted but no EmbeddingPort was provided"
            )
        entity_texts = [
            f"{entity.label}: {entity.text}" for entity in gliner_entities
        ]
        relation_texts = [
            f"{relation.source} {relation.relation} {relation.target} "
            f"{relation.fact or relation.description or ''}"
            for relation in gliner_relations
        ]
        entity_vectors = await embed_texts_in_batches(entity_texts, embedding_port)
        relation_vectors = await embed_texts_in_batches(
            relation_texts, embedding_port
        )
        gliner_entities = [
            entity.model_copy(update={"embedding": vector})
            for entity, vector in zip(gliner_entities, entity_vectors, strict=True)
        ]
        gliner_relations = [
            relation.model_copy(update={"embedding": vector})
            for relation, vector in zip(
                gliner_relations, relation_vectors, strict=True
            )
        ]

    site_payload = SitePayload(
        site_id=site_id,
        domain=domain,
        root_url=root_url,
        title=pages[0].title if pages else domain,
        summary=site_summary,
        gliner_metadata=gliner_metadata,
        entities=gliner_entities,
        relations=gliner_relations,
    )
    return CrawlIngestion(
        run_id=str(uuid4()),
        root_url=root_url,
        max_depth=max_depth,
        started_at=crawl_start,
        finished_at=crawl_finish,
        site=site_payload,
        pages=list(pages),
    )


async def ingest_crawl_documents(
    documents: Sequence[CrawlDocument],
    root_url: str,
    embedding_port: EmbeddingPort,
    text_processor: TextProcessor,
    graph_store: GraphStorePort,
    crawling_service: CrawlingService | None = None,
    gliner_port: GlinerPort | None = None,
    chat_generator: ChatGeneratorPort | None = None,
    settings: Any = None,
    max_depth: int = 3,
    chunk_size: int = 5000,
    lang_extract: Any = None,
) -> GraphOperationResult:
    """Ingest crawled documents into graph store via CrawlIngestion."""
    started_at = datetime.now(timezone.utc)
    pages: list[PagePayload] = []

    for doc in documents:
        if not doc.success or not doc.markdown:
            continue
        page = await build_page_payload(
            url=doc.url,
            markdown_content=doc.markdown,
            embedding_port=embedding_port,
            text_processor=text_processor,
            crawling_service=crawling_service,
            settings=settings,
            title=doc.title,
            status_code=doc.status_code,
            content_type=doc.content_type,
            language=doc.language,
            links=doc.links,
            lang_extract=lang_extract,
            chunk_size=chunk_size,
        )
        pages.append(page)

    if not pages:
        return GraphOperationResult(
            success=False,
            error="No valid pages to ingest",
        )

    finished_at = datetime.now(timezone.utc)
    ingestion = await build_crawl_ingestion(
        root_url=root_url,
        pages=pages,
        max_depth=max_depth,
        started_at=started_at,
        finished_at=finished_at,
        gliner_port=gliner_port,
        chat_generator=chat_generator,
        embedding_port=embedding_port,
        settings=settings,
    )

    return await graph_store.ingest_crawl(ingestion)
