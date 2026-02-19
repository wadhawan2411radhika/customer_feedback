"""Pydantic models for RAG system."""

from pydantic import BaseModel
from typing import Any


class SearchResult(BaseModel):
    """Result from semantic search with full metadata."""

    content: str
    feedback_summary: dict[str, Any]
    feedback_record: dict[str, Any]
    score: float


class RAGResponse(BaseModel):
    """Response from RAG pipeline."""

    answer: str
