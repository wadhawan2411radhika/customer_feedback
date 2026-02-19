"""Tests for src.rag_pipeline — constructor injection, no patching."""

from unittest.mock import AsyncMock

from src.models import SearchResult
from src.rag_pipeline import RAGPipeline


def _make_pipeline(search_results=None, llm_chunks=None):
    """Build RAGPipeline with mock retriever and llm_client."""
    mock_retriever = AsyncMock()
    mock_retriever.search = AsyncMock(return_value=search_results or [])

    mock_llm = AsyncMock()

    async def fake_generate(query, results):
        for chunk in (llm_chunks or []):
            yield chunk

    mock_llm.generate_answer = fake_generate

    pipeline = RAGPipeline(retriever=mock_retriever, llm_client=mock_llm)
    return pipeline, mock_retriever, mock_llm


class TestRAGPipelineQuery:
    async def test_with_results_streams_llm_chunks(self, sample_search_results):
        pipeline, _, _ = _make_pipeline(
            search_results=sample_search_results, llm_chunks=["chunk1", "chunk2"]
        )
        output = [c async for c in pipeline.query("test query")]
        assert output == ["chunk1", "chunk2"]

    async def test_no_results_yields_fallback_message(self):
        pipeline, _, _ = _make_pipeline(search_results=[], llm_chunks=[])
        output = [c async for c in pipeline.query("test query")]
        assert output == ["No relevant feedback found for your query."]

    async def test_calls_retriever_search_with_query(self, sample_search_results):
        pipeline, mock_retriever, _ = _make_pipeline(
            search_results=sample_search_results, llm_chunks=["x"]
        )
        [c async for c in pipeline.query("my question")]
        mock_retriever.search.assert_called_once_with("my question")

    async def test_does_not_call_llm_when_no_results(self):
        mock_retriever = AsyncMock()
        mock_retriever.search = AsyncMock(return_value=[])

        mock_llm = AsyncMock()
        call_tracker = []

        async def fake_generate(query, results):
            call_tracker.append(True)
            return
            yield

        mock_llm.generate_answer = fake_generate

        pipeline = RAGPipeline(retriever=mock_retriever, llm_client=mock_llm)
        [c async for c in pipeline.query("test")]
        assert call_tracker == []
