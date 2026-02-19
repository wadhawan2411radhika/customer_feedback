"""Semantic search retriever for feedback summaries."""

import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from src.embeddings import LocalEmbeddings
from src.models import SearchResult


class FeedbackRetriever:
    """Retrieves relevant feedback summaries using semantic search."""

    def __init__(
        self,
        index_path: Path = Path(".chroma_db"),
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        top_k: int = 5,
        api_key: str | None = None,
    ):
        self.index_path = Path(index_path)
        self.embedding_model_name = embedding_model
        self.embeddings_client = LocalEmbeddings(model=embedding_model)
        self.top_k = top_k
        self.client = chromadb.PersistentClient(
            path=str(self.index_path), settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="feedback_summaries",
            metadata={"embedding_model": embedding_model},
        )

    async def search(self, query: str) -> list[SearchResult]:
        """Search for relevant feedback summaries."""
        query_embedding = await self.embeddings_client.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
        )

        search_results = []

        if not results["ids"] or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i, doc_id in enumerate(ids):
            metadata = metadatas[i]
            content = documents[i]
            distance = distances[i]

            summary_json = json.loads(metadata["summary_json"])
            record_json = json.loads(metadata["record_json"])

            score = 1.0 - distance

            search_results.append(
                SearchResult(
                    content=content,
                    feedback_summary=summary_json,
                    feedback_record=record_json,
                    score=score,
                )
            )

        return search_results
