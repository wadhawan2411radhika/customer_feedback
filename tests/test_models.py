"""Tests for src.models — no mocking needed."""

import pytest
from pydantic import ValidationError

from src.models import RAGResponse, SearchResult


class TestSearchResult:
    def test_valid_construction(self):
        result = SearchResult(
            content="App crashes on upload",
            feedback_summary={"id": "s1"},
            feedback_record={"id": "r1"},
            score=0.95,
        )
        assert result.content == "App crashes on upload"
        assert result.score == 0.95

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            SearchResult(content="text", feedback_summary={}, score=0.5)

    def test_serialization_roundtrip(self):
        original = SearchResult(
            content="slow login",
            feedback_summary={"a": 1},
            feedback_record={"b": 2},
            score=0.8,
        )
        data = original.model_dump()
        restored = SearchResult(**data)
        assert restored == original


class TestRAGResponse:
    def test_valid_construction(self):
        resp = RAGResponse(answer="This is the answer.")
        assert resp.answer == "This is the answer."

    def test_missing_answer_raises(self):
        with pytest.raises(ValidationError):
            RAGResponse()

    def test_serialization_roundtrip(self):
        original = RAGResponse(answer="test answer")
        data = original.model_dump()
        restored = RAGResponse(**data)
        assert restored == original
