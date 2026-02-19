"""Vector index builder for feedback summaries."""

import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from src.embeddings import LocalEmbeddings


class FeedbackIndexer:
    """Indexes feedback summaries into ChromaDB with local embeddings."""

    def __init__(
        self,
        data_dir: Path,
        index_path: Path = Path(".chroma_db"),
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.index_path = Path(index_path)
        self.embedding_model_name = embedding_model
        self.embeddings_client = LocalEmbeddings(model=embedding_model, device=device)
        self.client = chromadb.PersistentClient(
            path=str(self.index_path), settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="feedback_summaries",
            metadata={"embedding_model": embedding_model},
        )

    def _extract_content(self, summary_data: dict[str, Any]) -> str | None:
        """Extract content from feedback_summary JSON."""
        content_attr = summary_data.get("attributes", {}).get("content", {})
        values = content_attr.get("string", {}).get("values", [])
        if values and values[0]:
            return values[0]
        return None

    def _load_record_pair(self, summary_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Load feedback_summary and feedback_record pair."""
        summary_file = self.data_dir / summary_id / "feedback_summary.json"
        record_file = self.data_dir / summary_id / "feedback_record.json"

        if not summary_file.exists() or not record_file.exists():
            return None

        with open(summary_file) as f:
            summary_data = json.load(f)

        with open(record_file) as f:
            record_data = json.load(f)

        return summary_data, record_data

    def _get_feedback_record_id(self, summary_data: dict[str, Any]) -> str | None:
        """Extract feedback_record_id from summary."""
        record_id_attr = summary_data.get("attributes", {}).get("feedback_record_id", {})
        values = record_id_attr.get("string", {}).get("values", [])
        if values and values[0]:
            return values[0]
        return None

    async def index_all(self) -> int:
        """Index all feedback summaries in data directory."""
        summary_dirs = [
            d for d in self.data_dir.iterdir() if d.is_dir() and (d / "feedback_summary.json").exists()
        ]

        documents = []
        metadatas = []
        ids = []

        for summary_dir in summary_dirs:
            summary_id = summary_dir.name
            pair = self._load_record_pair(summary_id)

            if not pair:
                continue

            summary_data, record_data = pair
            content = self._extract_content(summary_data)

            if not content:
                continue

            record_id = self._get_feedback_record_id(summary_data)

            documents.append(content)
            metadatas.append(
                {
                    "summary_id": summary_id,
                    "record_id": record_id or "",
                    "summary_json": json.dumps(summary_data),
                    "record_json": json.dumps(record_data),
                }
            )
            ids.append(summary_id)

        if not documents:
            return 0

        embeddings = await self.embeddings_client.embed_texts(documents)

        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

        return len(documents)