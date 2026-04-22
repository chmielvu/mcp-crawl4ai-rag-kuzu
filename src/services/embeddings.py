"""Embeddings and chat service backed by Mistral AI."""

import asyncio
import logging
from typing import Any, Callable, List, Optional, Tuple, cast

from mistralai.client import Mistral

from crawl4ai_mcp.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for creating embeddings and text completions."""

    def __init__(self, settings: Optional[Any] = None):
        self.settings = settings or get_settings()
        self.client = Mistral(api_key=self.settings.mistral_api_key)
        self.max_retries = 3
        self.retry_delay = 1.0

    async def _run_in_executor(self, func: Callable[[], Any]) -> Any:
        """Run a synchronous function in a worker thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func)

    async def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts."""
        if not texts:
            return []

        for retry in range(self.max_retries):
            try:
                def create_batch_embeddings() -> Any:
                    return self.client.embeddings.create(
                        model=self.settings.embedding_model,
                        inputs=texts,
                    )

                response = await self._run_in_executor(create_batch_embeddings)
                return [item.embedding for item in response.data]
            except Exception as error:
                if "rate_limit" in str(error).lower() and retry < self.max_retries - 1:
                    wait_time = self.retry_delay * (2**retry)
                    logger.warning("Rate limit hit, retrying in %s seconds", wait_time)
                    await asyncio.sleep(wait_time)
                    continue

                if retry == self.max_retries - 1:
                    logger.error(
                        "Batch embedding failed after %s retries; falling back to single requests",
                        self.max_retries,
                    )
                    return await self._create_embeddings_individually(texts)

        return [[0.0] * self.settings.embedding_dimensions for _ in texts]

    async def _create_embeddings_individually(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for index, text in enumerate(texts):
            try:
                def create_single_embedding() -> Any:
                    return self.client.embeddings.create(
                        model=self.settings.embedding_model,
                        inputs=[text],
                    )

                response = await self._run_in_executor(create_single_embedding)
                embeddings.append(response.data[0].embedding)
                if index < len(texts) - 1:
                    await asyncio.sleep(0.1)
            except Exception as error:
                logger.error("Embedding failed for item %s: %s", index, error)
                embeddings.append([0.0] * self.settings.embedding_dimensions)
        return embeddings

    async def create_embedding(self, text: str) -> List[float]:
        """Create an embedding for a single text."""
        embeddings = await self.create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * self.settings.embedding_dimensions

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 150,
    ) -> str:
        """Generate a text completion using the configured Mistral model."""
        def complete_chat() -> Any:
            return self.client.chat.complete(
                model=self.settings.model_choice,
                messages=cast(Any, messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )

        response = await self._run_in_executor(complete_chat)
        return _extract_message_text(response)

    async def generate_contextual_embedding(
        self,
        full_document: str,
        chunk: str,
    ) -> Tuple[str, bool]:
        """Generate short contextual text for a chunk."""
        prompt = f"""<document>
{full_document[:25000]}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the whole document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
        messages = [
            {
                "role": "system",
                "content": "You provide concise contextual information for retrieval.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            context = await self.chat_complete(messages, temperature=0.0, max_tokens=200)
            return f"{context}\n---\n{chunk}", True
        except Exception as error:
            logger.error("Contextual embedding generation failed: %s", error)
            return chunk, False

    async def process_chunks_with_context(
        self,
        chunks: List[Tuple[str, str, str]],
        max_workers: int = 10,
    ) -> List[Tuple[str, bool]]:
        """Process multiple chunks concurrently."""
        del max_workers
        tasks = [
            self.generate_contextual_embedding(full_document, content)
            for _, content, full_document in chunks
        ]
        results: list[Tuple[str, bool] | BaseException] = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        processed: List[Tuple[str, bool]] = []
        for index, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Chunk contextualization failed for item %s: %s", index, result)
                processed.append((chunks[index][1], False))
            else:
                processed.append(cast(Tuple[str, bool], result))
        return processed

    def process_chunk_with_context(self, args: Tuple[str, str, str]) -> Tuple[str, bool]:
        """Synchronous wrapper for contextual chunk processing."""
        _, content, full_document = args
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(
            self.generate_contextual_embedding(full_document, content)
        )


def _extract_message_text(response: Any) -> str:
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts).strip()
    return str(content).strip()
