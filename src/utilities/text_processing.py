"""Text processing utilities for chunking, context generation, and content extraction."""

from __future__ import annotations

import logging
import re
from typing import Any

from crawl4ai_mcp.services.contracts import ChatGeneratorPort

logger = logging.getLogger(__name__)


class TextProcessor:
    """Utility class for text processing and semantic chunking."""

    def __init__(
        self,
        settings: Any,
        chat_generator: ChatGeneratorPort,
    ) -> None:
        """Initialize the text processor with the chat-only provider."""
        self.settings = settings
        self.chat_generator = chat_generator

    def smart_chunk_markdown(self, text: str, chunk_size: int = 5000) -> list[str]:
        """Split text into chunks while trying to preserve semantic boundaries."""
        chunks: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            if end >= text_length:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
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

    def smart_chunk_with_offsets(
        self, text: str, chunk_size: int = 5000
    ) -> list[dict[str, Any]]:
        """Split text into chunks with start/end character offsets and heading hierarchy."""
        if not text:
            return []

        chunks_meta: list[dict[str, Any]] = []
        start = 0
        text_length = len(text)
        index = 0

        while start < text_length:
            end = start + chunk_size
            if end >= text_length:
                raw_chunk = text[start:]
                chunk_text = raw_chunk.strip()
                if chunk_text:
                    rel_start = raw_chunk.find(chunk_text)
                    actual_start = start + (rel_start if rel_start != -1 else 0)
                    actual_end = actual_start + len(chunk_text)
                    heading_path = _compute_heading_path(text, actual_start, actual_end)
                    section_info = self.extract_section_info(chunk_text)
                    chunks_meta.append(
                        {
                            "index": index,
                            "text": chunk_text,
                            "start_char": actual_start,
                            "end_char": actual_end,
                            "heading_path": heading_path,
                            "section_info": section_info,
                        }
                    )
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

            raw_chunk = text[start:end]
            chunk_text = raw_chunk.strip()
            if chunk_text:
                rel_start = raw_chunk.find(chunk_text)
                actual_start = start + (rel_start if rel_start != -1 else 0)
                actual_end = actual_start + len(chunk_text)
                heading_path = _compute_heading_path(text, actual_start, actual_end)
                section_info = self.extract_section_info(chunk_text)
                chunks_meta.append(
                    {
                        "index": index,
                        "text": chunk_text,
                        "start_char": actual_start,
                        "end_char": actual_end,
                        "heading_path": heading_path,
                        "section_info": section_info,
                    }
                )
                index += 1
            start = end

        return chunks_meta

    def extract_section_info(self, chunk: str) -> dict[str, Any]:
        """Extract headings and basic stats from a markdown chunk."""
        headers = re.findall(r"^(#+)\s+(.+)$", chunk, re.MULTILINE)
        header_str = (
            "; ".join(f"{level} {title}" for level, title in headers) if headers else ""
        )
        return {
            "headers": header_str,
            "char_count": len(chunk),
            "word_count": len(chunk.split()),
        }

    async def generate_contextual_embedding(
        self,
        full_document: str,
        chunk: str,
    ) -> tuple[str, bool]:
        """Generate short context that situates a chunk inside a larger document."""
        if self.chat_generator is None:
            return chunk, False

        prompt = f"""<document>
{full_document[:25000]}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that provides concise "
                    "contextual information."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            context = await self.chat_generator.chat_complete(
                messages=messages,
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
    ) -> tuple[str, bool]:
        """Async wrapper for generating contextualized chunk content."""
        del url
        return await self.generate_contextual_embedding(full_document, content)

    async def process_code_example(
        self,
        code: str,
        context_before: str,
        context_after: str,
    ) -> str:
        """Generate a summary using the injected chat-only provider."""

        return await self.chat_generator.generate_code_example_summary(
            code, context_before, context_after
        )


def _compute_heading_path(text: str, start_char: int, end_char: int) -> str:
    """Determine hierarchical heading path for text range."""
    headings = list(re.finditer(r"^(#+)\s+(.+)$", text[:end_char], re.MULTILINE))
    if not headings:
        return ""

    stack: list[tuple[int, str]] = []
    for match in headings:
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

    return " > ".join(title for _, title in stack)
