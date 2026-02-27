"""OpenAI LLM client for answer generation."""

import json
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

#     def _build_prompt(self, query: str, search_results: list[SearchResult]) -> str:
#         """Build context-aware prompt from query and retrieved results."""
#         context_parts = []

#         for i, result in enumerate(search_results, 1):
#             context_parts.append(f"Feedback {i}: {result.content}")

#         context = "\n\n".join(context_parts)

#         prompt = f"""You are a helpful assistant that answers questions based on user feedback summaries.

# Context from user feedback:
# {context}

# Question: {query}

# Based on the feedback summaries above, provide a comprehensive answer to the question. If the feedback doesn't contain relevant information, say so clearly.

# Answer:"""

#         return prompt

    def _build_prompt(self, query: str, search_results: list[SearchResult]) -> str:
        """Build context-aware prompt with raw feedback for inline quote extraction."""
        context_parts = []

        for i, result in enumerate(search_results, 1):
            # Extract from nested structure per the data schema
            context_parts.append(
                f"[Source {i}]\n"
                f"feedback_record_id: {result.feedback_summary['attributes']['feedback_record_id']['string']['values'][0]}\n"
                f"Summary: {result.content}\n"
                f"Original feedback (verbatim): {result.feedback_record['attributes']['content']['string']['values'][0]}"
            )

        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a product insights analyst answering questions based on user feedback data.

        You have access to two types of context for each source:
        - A **summary** (paraphrased overview of the feedback)
        - A **verbatim feedback** (exact original user text inside <verbatim_feedback> tags)

        Context:
        {context}

        ---

        Question: {query}

        Instructions:

        Summary instructions:
        1. Based on the feedback summaries above, provide a comprehensive answer to the question. If the feedback doesn't contain relevant information, say so clearly.
        2. Write a fluent, coherent answer using the summaries as your structural guide.

        Quote instructions:
        1. After every claim from summary, immediately support it with a verbatim quote from the <verbatim_feedback> tag of the relevant source.
        2. Quotes MUST be formatted EXACTLY like this — a markdown block quote followed by the record ID:

        > "<verbatim_feedback" — feedback_record_id

        3. STRICT RULES for quotes:
        - The quote must be a verbatim substring of the text inside followed by ">" tag — do NOT paraphrase
        - Do NOT fabricate quotes or record IDs
        - Do NOT quote from the summary — only from <verbatim_feedback>
        - Do NOT collect all quotes at the end — each must follow the claim it supports in the following line
        4. One claim can be supported by multiple quotes from different records — add each on its own line followed by ">" tag
        5. If no verbatim text supports a claim, state it without a quote rather than fabricating one.
        6. Verbatim quotes should be in the next line after summary

        Feedback record id instructions:
        1. Each inline quote shall be supported by feedback record id

        ## OUTPUT FORMAT:

        Summary followed by ">" tags followed by "<verbatim quotes" followed by feedback record id
        

        Example of output format:

        Users frequently complain about performance degradation after updates.
        > "the app was much better before the update, now it freezes constantly" — rec_abc123
        > "every time I update it gets slower and slower" — rec_def456

        Answer:"""

        return prompt

    async def generate_answer(
        self, query: str, search_results: list[SearchResult]
    ) -> AsyncGenerator[str, None]:
        """Stream answer chunks from query and retrieved context."""
        prompt = self._build_prompt(query, search_results)
        print("Tokens of prompt: ", len(prompt)//4)
        # def save_results(results: list[SearchResult], path: str = "search_results.json"):
        #     with open(path, "w") as f:
        #         json.dump([r.model_dump() for r in results], f, indent=2)
        # save_results(search_results)
        
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
