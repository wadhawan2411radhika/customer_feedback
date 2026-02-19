"""Groq LLM client for answer generation."""

import os
from collections.abc import AsyncGenerator

from groq import AsyncGroq

from src.models import SearchResult


class GroqClient:
    """Async Groq client for generating answers from retrieved context.

    Free models available on Groq:
        - "qwen/qwen3-32b"         (best quality, supports reasoning)
        - "llama-3.3-70b-versatile" (strong general purpose)
        - "llama-3.1-8b-instant"   (fastest, lightweight)
        - "gemma2-9b-it"           (good for instruction following)

    Get a free API key at: https://console.groq.com
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen/qwen3-32b",
        temperature: float = 0.6,
        max_completion_tokens: int = 4096,
        top_p: float = 0.95,
        reasoning_effort: str = "default",
    ):
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")

        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort

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
            max_completion_tokens=self.max_completion_tokens,
            top_p=self.top_p,
            reasoning_effort=self.reasoning_effort,
            stream=True,
            stop=None,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content