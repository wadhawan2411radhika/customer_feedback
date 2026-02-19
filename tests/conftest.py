"""Shared fixtures for all test modules."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.models import SearchResult

SAMPLE_SUMMARY_DATA = {
    "id": "summary-001",
    "attributes": {
        "content": {
            "string": {
                "values": ["The app keeps crashing when I try to upload photos."]
            }
        },
        "feedback_record_id": {
            "string": {
                "values": ["record-001"]
            }
        },
    },
}

SAMPLE_RECORD_DATA = {
    "id": "record-001",
    "attributes": {
        "source": {"string": {"values": ["zendesk"]}},
        "created_at": {"string": {"values": ["2024-01-15T10:00:00Z"]}},
    },
}

SAMPLE_EMBEDDING = [0.1] * 1536


@pytest.fixture
def sample_summary_data():
    return SAMPLE_SUMMARY_DATA.copy()


@pytest.fixture
def sample_record_data():
    return SAMPLE_RECORD_DATA.copy()


@pytest.fixture
def sample_embedding():
    return SAMPLE_EMBEDDING.copy()


@pytest.fixture
def sample_search_results():
    return [
        SearchResult(
            content="The app crashes on photo upload.",
            feedback_summary=SAMPLE_SUMMARY_DATA,
            feedback_record=SAMPLE_RECORD_DATA,
            score=0.92,
        ),
        SearchResult(
            content="Login page is very slow.",
            feedback_summary={"id": "summary-002", "attributes": {}},
            feedback_record={"id": "record-002", "attributes": {}},
            score=0.85,
        ),
    ]


@pytest.fixture
def mock_openai_embeddings_client():
    """AsyncMock of AsyncOpenAI returning fake embeddings."""
    client = AsyncMock()
    embedding_obj = MagicMock()
    embedding_obj.embedding = SAMPLE_EMBEDDING
    response = MagicMock()
    response.data = [embedding_obj]
    client.embeddings.create = AsyncMock(return_value=response)
    return client


@pytest.fixture
def mock_chroma_collection():
    """MagicMock ChromaDB collection with pre-configured add/query."""
    collection = MagicMock()
    collection.add = MagicMock()
    collection.query = MagicMock(
        return_value={
            "ids": [["id1", "id2"]],
            "documents": [["doc content 1", "doc content 2"]],
            "metadatas": [
                [
                    {
                        "summary_id": "id1",
                        "record_id": "rec1",
                        "summary_json": json.dumps(SAMPLE_SUMMARY_DATA),
                        "record_json": json.dumps(SAMPLE_RECORD_DATA),
                    },
                    {
                        "summary_id": "id2",
                        "record_id": "rec2",
                        "summary_json": json.dumps({"id": "summary-002"}),
                        "record_json": json.dumps({"id": "record-002"}),
                    },
                ]
            ],
            "distances": [[0.1, 0.2]],
        }
    )
    return collection
