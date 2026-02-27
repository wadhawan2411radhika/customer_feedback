"""OpenAI LLM client for answer generation."""

import json
import os
from collections.abc import AsyncGenerator
import time
from typing import Any

from openai import AsyncOpenAI
from src.cost_tracker import CostRecord
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

    def _build_baseline_prompt(self, query: str, search_results: list[SearchResult]) -> str:
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

    def _build_enhanced_prompt(self, query: str, search_results: list[SearchResult]) -> str:
        """Build prompt that produces prose + attributed block quotes."""
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            record_id = result.feedback_summary['attributes']['feedback_record_id']['string']['values'][0]
            raw_content = result.feedback_record['attributes']['content']['string']['values'][0]
            summary = result.content

            context_parts.append(
                f"[RECORD {i}]\n"
                f"feedback_record_id: {record_id}\n"
                f"summary: {summary}\n"
                f"verbatim_feedback: {raw_content}"
            )

        context = "\n\n".join(context_parts)

        prompt = f"""You are a product insights analyst. Answer the question below using the provided user feedback.

    ## Feedback Records
    {context}

    ## Question
    {query}

    ## Instructions

    **Prose:** Write a fluent, paragraph-form answer synthesizing the summaries. 
    Group related points together — do not write one sentence per record.

    **Quotes:** After each prose paragraph, add verbatim block quotes that support the claims in that paragraph.

    Rules for quotes:
    - Copy text EXACTLY from `verbatim_feedback` — not from `summary`
    - You may quote a substring, but it must match character-for-character
    - Format every quote on its own line like this:
    > "exact text copied from verbatim_feedback" — feedback_record_id
    - If a paragraph is supported by multiple records, add one quote per line
    - Never fabricate a quote or record ID
    - If no verbatim text supports a claim, omit the quote rather than inventing one

    ## Output Format

    Write in this repeating structure:

    Paragraph making one or more related claims from the feedback.

    > "verbatim quote supporting a claim above" — feedback_record_id
    > "another verbatim quote if relevant" — feedback_record_id

    Next paragraph making different claims.

    > "verbatim quote" — feedback_record_id

    Answer:"""

        return prompt

    async def generate_answer(self, query: str, search_results: list[SearchResult], mode="enhanced") -> AsyncGenerator[str, None]:
        
        if mode=="enhanced":
            prompt = self._build_enhanced_prompt(query, search_results)
        else:
            prompt = self._build_baseline_prompt(query, search_results)
        self.last_cost = CostRecord(model=self.model, version=mode, query=query)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes user feedback."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            stream=True,
            stream_options={"include_usage": True},
        )

        t_start = time.perf_counter()
        first_chunk = True

        async for chunk in stream:
            if chunk.usage:
                self.last_cost.input_tokens = chunk.usage.prompt_tokens
                self.last_cost.output_tokens = chunk.usage.completion_tokens
                self.last_cost.total_time = time.perf_counter() - t_start
            if chunk.choices and chunk.choices[0].delta.content:
                if first_chunk:
                    self.last_cost.time_to_first_token = time.perf_counter() - t_start
                    first_chunk = False
                yield chunk.choices[0].delta.content
