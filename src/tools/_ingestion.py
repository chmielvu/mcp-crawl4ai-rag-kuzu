"""Shared ingestion helpers for crawl tools."""

from typing import Any
from urllib.parse import urlparse

from crawl4ai_mcp.config import Settings
from crawl4ai_mcp.services.crawling import CrawlingService
from crawl4ai_mcp.services.database import DatabaseService
from crawl4ai_mcp.services.embeddings import EmbeddingService
from crawl4ai_mcp.utilities.metadata import create_chunk_metadata
from crawl4ai_mcp.utilities.text_processing import TextProcessor


async def update_source_records(
    results: list[dict[str, str]],
    database_service: DatabaseService,
    crawling_service: CrawlingService,
) -> list[str]:
    source_ids = sorted({urlparse(result["url"]).netloc for result in results})
    for source_id in source_ids:
        source_content = "\n\n".join(
            result["markdown"]
            for result in results
            if urlparse(result["url"]).netloc == source_id
        )
        source_summary = await crawling_service.extract_source_summary(
            source_id,
            source_content[:10000],
        )
        await database_service.update_source_info(
            source_id=source_id,
            summary=source_summary,
            word_count=len(source_content.split()),
        )
    return source_ids


async def ingest_markdown_result(
    result_url: str,
    markdown_content: str,
    crawl_type: str,
    settings: Settings,
    embedding_service: EmbeddingService,
    database_service: DatabaseService,
    crawling_service: CrawlingService,
    text_processor: TextProcessor,
    chunk_size: int,
) -> tuple[int, int]:
    source_id = urlparse(result_url).netloc
    chunks = text_processor.smart_chunk_markdown(markdown_content, chunk_size=chunk_size)
    if not chunks:
        return 0, 0

    document_payload = await _build_document_payload(
        result_url=result_url,
        markdown_content=markdown_content,
        crawl_type=crawl_type,
        source_id=source_id,
        chunks=chunks,
        settings=settings,
        embedding_service=embedding_service,
        text_processor=text_processor,
    )
    document_result = await database_service.add_documents(**document_payload)

    code_count = 0
    if settings.use_agentic_rag:
        code_payload = await _build_code_payload(
            result_url=result_url,
            markdown_content=markdown_content,
            embedding_service=embedding_service,
            crawling_service=crawling_service,
        )
        if code_payload:
            code_result = await database_service.add_code_examples(**code_payload)
            code_count = code_result.get("count", 0)

    return document_result.get("count", 0), code_count


async def _build_document_payload(
    result_url: str,
    markdown_content: str,
    crawl_type: str,
    source_id: str,
    chunks: list[str],
    settings: Settings,
    embedding_service: EmbeddingService,
    text_processor: TextProcessor,
) -> dict[str, Any]:
    urls: list[str] = []
    chunk_numbers: list[int] = []
    contents: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []

    for index, chunk in enumerate(chunks, start=1):
        urls.append(result_url)
        chunk_numbers.append(index)
        contents.append(chunk)

        if settings.use_contextual_embeddings:
            contextual_content, _ = await text_processor.generate_contextual_embedding(
                markdown_content,
                chunk,
            )
            embedding = await embedding_service.create_embedding(contextual_content)
        else:
            embedding = await embedding_service.create_embedding(chunk)
        embeddings.append(embedding)

        section_info = text_processor.extract_section_info(chunk)
        metadatas.append(
            create_chunk_metadata(
                chunk=chunk,
                source_id=source_id,
                url=result_url,
                chunk_index=index - 1,
                crawl_type=crawl_type,
                section_info=section_info,
            )
        )

    return {
        "urls": urls,
        "chunk_numbers": chunk_numbers,
        "contents": contents,
        "embeddings": embeddings,
        "metadatas": metadatas,
        "url_to_full_document": {result_url: markdown_content},
    }


async def _build_code_payload(
    result_url: str,
    markdown_content: str,
    embedding_service: EmbeddingService,
    crawling_service: CrawlingService,
) -> dict[str, Any] | None:
    code_blocks = crawling_service.extract_code_blocks(markdown_content)
    if not code_blocks:
        return None

    urls: list[str] = []
    chunk_numbers: list[int] = []
    code_examples: list[str] = []
    summaries: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict] = []

    for index, code_block in enumerate(code_blocks, start=1):
        formatted_code = _format_code_block(code_block["language"], code_block["code"])
        summary = await crawling_service.generate_code_example_summary(
            code_block["code"],
            code_block["context_before"],
            code_block["context_after"],
        )
        urls.append(result_url)
        chunk_numbers.append(index)
        code_examples.append(formatted_code)
        summaries.append(summary)
        embeddings.append(
            await embedding_service.create_embedding(f"{summary}\n\n{formatted_code}")
        )
        metadatas.append(
            {
                "language": code_block["language"] or "unknown",
                "char_count": len(code_block["code"]),
                "summary": summary,
            }
        )

    return {
        "urls": urls,
        "chunk_numbers": chunk_numbers,
        "code_examples": code_examples,
        "summaries": summaries,
        "embeddings": embeddings,
        "metadatas": metadatas,
    }


def _format_code_block(language: str, code: str) -> str:
    if language:
        return f"```{language}\n{code}\n```"
    return f"```\n{code}\n```"
