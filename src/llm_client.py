"""OpenAI LLM client for answer generation."""

import os
from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from src.models import SearchResult


class OpenAIClient:
    """Async OpenAI client for generating answers from retrieved context."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def _build_prompt(self, query: str, search_results: list[SearchResult]) -> str:
        """Build context-aware prompt from query and retrieved results."""
        context_parts = []

        for i, result in enumerate(search_results, 1):
            context_parts.append(f"Feedback {i}: {result.content}")

        context = "\n\n".join(context_parts)

        prompt = f"""You are a helpful assistant that answers questions based on user feedback summaries.

Context from user feedback:
{context}

Question: {query}

Based on the feedback summaries above, provide a comprehensive answer to the question. If the feedback doesn't contain relevant information, say so clearly.

Answer:"""

        return prompt

    async def generate_answer(
        self, query: str, search_results: list[SearchResult]
    ) -> AsyncGenerator[str, None]:
        """Stream answer chunks from query and retrieved context."""
        prompt = self._build_prompt(query, search_results)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes user feedback."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
