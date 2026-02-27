"""RAG pipeline orchestrator."""

from collections.abc import AsyncGenerator

from src.llm_client import OpenAIClient
from src.retriever import FeedbackRetriever


class RAGPipeline:
    """Orchestrates retrieval and generation for RAG system."""

    def __init__(
        self,
        retriever: FeedbackRetriever,
        llm_client: OpenAIClient,
    ):
        self.retriever = retriever
        self.llm_client = llm_client

    async def query(self, query: str, mode: str = "enhanced") -> AsyncGenerator[str, None]:
        """Process query through RAG pipeline, streaming the answer."""
        search_results = await self.retriever.search(query)

        if not search_results:
            yield "No relevant feedback found for your query."
            return

        async for chunk in self.llm_client.generate_answer(query, search_results, mode=mode):
            yield chunk
