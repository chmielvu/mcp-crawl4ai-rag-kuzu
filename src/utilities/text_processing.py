"""Text processing utilities for chunking, context generation, and content extraction."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from crawl4ai_mcp.config import get_settings
from crawl4ai_mcp.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class TextProcessor:
    """Utility class for text processing operations."""

    def __init__(
        self,
        settings: Optional[Any] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """Initialize the text processor."""
        self.settings = settings or get_settings()
        self.embedding_service = embedding_service or EmbeddingService(self.settings)

    def smart_chunk_markdown(self, text: str, chunk_size: int = 5000) -> List[str]:
        """Split text into chunks while trying to preserve semantic boundaries."""
        chunks: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            if end >= text_length:
                chunks.append(text[start:].strip())
                break

            chunk = text[start:end]
            code_block = chunk.rfind("```")
            if code_block != -1 and code_block > chunk_size * 0.3:
                end = start + code_block
            elif "\n\n" in chunk:
                last_break = chunk.rfind("\n\n")
                if last_break > chunk_size * 0.3:
                    end = start + last_break
            elif ". " in chunk:
                last_period = chunk.rfind(". ")
                if last_period > chunk_size * 0.3:
                    end = start + last_period + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end

        return chunks

    def extract_section_info(self, chunk: str) -> Dict[str, Any]:
        """Extract headings and basic stats from a markdown chunk."""
        headers = re.findall(r"^(#+)\s+(.+)$", chunk, re.MULTILINE)
        header_str = "; ".join(f"{level} {title}" for level, title in headers) if headers else ""
        return {
            "headers": header_str,
            "char_count": len(chunk),
            "word_count": len(chunk.split()),
        }

    async def generate_contextual_embedding(
        self,
        full_document: str,
        chunk: str,
    ) -> Tuple[str, bool]:
        """Generate short context that situates a chunk inside a larger document."""
        try:
            prompt = f"""<document>
{full_document[:25000]}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
            context = await self.embedding_service.chat_complete(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant that provides concise "
                            "contextual information."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            return f"{context}\n---\n{chunk}", True
        except Exception as error:
            logger.error(
                "Error generating contextual embedding: %s. Using original chunk instead.",
                error,
            )
            return chunk, False

    async def process_chunk_with_context(
        self,
        url: str,
        content: str,
        full_document: str,
    ) -> Tuple[str, bool]:
        """Async wrapper for generating contextualized chunk content."""
        del url
        return await self.generate_contextual_embedding(full_document, content)

    async def process_code_example(
        self,
        code: str,
        context_before: str,
        context_after: str,
    ) -> str:
        """Generate a summary for a code example with surrounding context."""
        from crawl4ai_mcp.services.crawling import CrawlingService

        crawling_service = CrawlingService(
            crawler=None,
            settings=self.settings,
            embedding_service=self.embedding_service,
        )
        return await crawling_service.generate_code_example_summary(
            code,
            context_before,
            context_after,
        )
