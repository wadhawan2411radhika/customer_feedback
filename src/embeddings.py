"""OpenAI embeddings client for generating text embeddings."""

import asyncio
import os

from openai import AsyncOpenAI


class OpenAIEmbeddings:
    """Async OpenAI client for generating text embeddings."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        max_concurrent: int = 10,
        batch_size: int = 2048,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.batch_size = batch_size

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch of texts with semaphore control."""
        async with self.semaphore:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            return [item.embedding for item in response.data]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts with semaphore-controlled concurrency."""
        if not texts:
            return []

        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        tasks = [self._embed_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks)

        embeddings = []
        for batch_result in batch_results:
            embeddings.extend(batch_result)

        return embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=[query],
        )
        return response.data[0].embedding
