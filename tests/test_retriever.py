"""Tests for src.retriever — mock OpenAIEmbeddings and chromadb."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import SAMPLE_EMBEDDING, SAMPLE_RECORD_DATA, SAMPLE_SUMMARY_DATA


def _make_retriever(mock_query_return=None):
    """Construct FeedbackRetriever with mocked dependencies."""
    with patch("src.retriever.OpenAIEmbeddings") as mock_emb, patch(
        "src.retriever.chromadb.PersistentClient"
    ) as mock_chroma:
        mock_collection = MagicMock()
        if mock_query_return is not None:
            mock_collection.query.return_value = mock_query_return
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_emb_instance = AsyncMock()
        mock_emb_instance.embed_query = AsyncMock(return_value=SAMPLE_EMBEDDING)
        mock_emb.return_value = mock_emb_instance

        from src.retriever import FeedbackRetriever

        retriever = FeedbackRetriever(index_path="/tmp/test_chroma", api_key="sk-test")
        return retriever, mock_collection, mock_emb_instance


class TestSearch:
    async def test_returns_search_results_with_correct_fields(self):
        query_return = {
            "ids": [["id1"]],
            "documents": [["App crashes"]],
            "metadatas": [
                [
                    {
                        "summary_id": "id1",
                        "record_id": "rec1",
                        "summary_json": json.dumps(SAMPLE_SUMMARY_DATA),
                        "record_json": json.dumps(SAMPLE_RECORD_DATA),
                    }
                ]
            ],
            "distances": [[0.15]],
        }
        retriever, _, _ = _make_retriever(query_return)
        results = await retriever.search("crash")

        assert len(results) == 1
        assert results[0].content == "App crashes"
        assert results[0].feedback_summary == SAMPLE_SUMMARY_DATA
        assert results[0].feedback_record == SAMPLE_RECORD_DATA

    async def test_score_computed_as_one_minus_distance(self):
        query_return = {
            "ids": [["id1"]],
            "documents": [["text"]],
            "metadatas": [
                [
                    {
                        "summary_json": json.dumps({}),
                        "record_json": json.dumps({}),
                    }
                ]
            ],
            "distances": [[0.3]],
        }
        retriever, _, _ = _make_retriever(query_return)
        results = await retriever.search("test")
        assert abs(results[0].score - 0.7) < 1e-6

    async def test_empty_results_returns_empty_list(self):
        query_return = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        retriever, _, _ = _make_retriever(query_return)
        results = await retriever.search("nothing")
        assert results == []

    async def test_multiple_results_returned_in_order(self):
        query_return = {
            "ids": [["id1", "id2"]],
            "documents": [["first doc", "second doc"]],
            "metadatas": [
                [
                    {
                        "summary_json": json.dumps({"id": "s1"}),
                        "record_json": json.dumps({"id": "r1"}),
                    },
                    {
                        "summary_json": json.dumps({"id": "s2"}),
                        "record_json": json.dumps({"id": "r2"}),
                    },
                ]
            ],
            "distances": [[0.1, 0.2]],
        }
        retriever, _, _ = _make_retriever(query_return)
        results = await retriever.search("query")

        assert len(results) == 2
        assert results[0].content == "first doc"
        assert results[1].content == "second doc"
        assert results[0].score > results[1].score

    async def test_embed_query_called_with_query_string(self):
        query_return = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        retriever, _, mock_emb = _make_retriever(query_return)
        await retriever.search("my query")
        mock_emb.embed_query.assert_called_once_with("my query")

    async def test_collection_query_called_with_top_k(self):
        query_return = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        retriever, mock_collection, _ = _make_retriever(query_return)
        await retriever.search("test")
        mock_collection.query.assert_called_once_with(
            query_embeddings=[SAMPLE_EMBEDDING], n_results=retriever.top_k
        )
