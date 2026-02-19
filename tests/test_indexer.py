"""Tests for src.indexer — mock OpenAIEmbeddings and chromadb."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import SAMPLE_EMBEDDING, SAMPLE_RECORD_DATA, SAMPLE_SUMMARY_DATA


def _make_indexer(data_dir, index_path):
    """Construct FeedbackIndexer with mocked dependencies."""
    with patch("src.indexer.OpenAIEmbeddings") as mock_emb, patch(
        "src.indexer.chromadb.PersistentClient"
    ) as mock_chroma:
        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection

        mock_emb_instance = AsyncMock()
        mock_emb_instance.embed_texts = AsyncMock(return_value=[SAMPLE_EMBEDDING])
        mock_emb.return_value = mock_emb_instance

        from src.indexer import FeedbackIndexer

        indexer = FeedbackIndexer(
            data_dir=data_dir,
            index_path=index_path,
            api_key="sk-test",
        )
        return indexer, mock_collection, mock_emb_instance


class TestExtractContent:
    def test_valid_data(self, sample_summary_data):
        with patch("src.indexer.OpenAIEmbeddings"), patch(
            "src.indexer.chromadb.PersistentClient"
        ):
            from src.indexer import FeedbackIndexer

            indexer = FeedbackIndexer.__new__(FeedbackIndexer)
            result = indexer._extract_content(sample_summary_data)
            assert result == "The app keeps crashing when I try to upload photos."

    def test_missing_attributes(self):
        with patch("src.indexer.OpenAIEmbeddings"), patch(
            "src.indexer.chromadb.PersistentClient"
        ):
            from src.indexer import FeedbackIndexer

            indexer = FeedbackIndexer.__new__(FeedbackIndexer)
            assert indexer._extract_content({}) is None

    def test_empty_values(self):
        data = {"attributes": {"content": {"string": {"values": [""]}}}}
        with patch("src.indexer.OpenAIEmbeddings"), patch(
            "src.indexer.chromadb.PersistentClient"
        ):
            from src.indexer import FeedbackIndexer

            indexer = FeedbackIndexer.__new__(FeedbackIndexer)
            assert indexer._extract_content(data) is None


class TestGetFeedbackRecordId:
    def test_valid_data(self, sample_summary_data):
        with patch("src.indexer.OpenAIEmbeddings"), patch(
            "src.indexer.chromadb.PersistentClient"
        ):
            from src.indexer import FeedbackIndexer

            indexer = FeedbackIndexer.__new__(FeedbackIndexer)
            result = indexer._get_feedback_record_id(sample_summary_data)
            assert result == "record-001"

    def test_missing_field(self):
        with patch("src.indexer.OpenAIEmbeddings"), patch(
            "src.indexer.chromadb.PersistentClient"
        ):
            from src.indexer import FeedbackIndexer

            indexer = FeedbackIndexer.__new__(FeedbackIndexer)
            assert indexer._get_feedback_record_id({}) is None


class TestLoadRecordPair:
    def test_both_files_present(self, tmp_path):
        sid = "test-id"
        d = tmp_path / sid
        d.mkdir()
        (d / "feedback_summary.json").write_text(json.dumps(SAMPLE_SUMMARY_DATA))
        (d / "feedback_record.json").write_text(json.dumps(SAMPLE_RECORD_DATA))

        indexer, _, _ = _make_indexer(tmp_path, tmp_path / ".chroma")
        result = indexer._load_record_pair(sid)
        assert result is not None
        assert result[0] == SAMPLE_SUMMARY_DATA
        assert result[1] == SAMPLE_RECORD_DATA

    def test_summary_missing(self, tmp_path):
        sid = "test-id"
        d = tmp_path / sid
        d.mkdir()
        (d / "feedback_record.json").write_text(json.dumps(SAMPLE_RECORD_DATA))

        indexer, _, _ = _make_indexer(tmp_path, tmp_path / ".chroma")
        assert indexer._load_record_pair(sid) is None

    def test_record_missing(self, tmp_path):
        sid = "test-id"
        d = tmp_path / sid
        d.mkdir()
        (d / "feedback_summary.json").write_text(json.dumps(SAMPLE_SUMMARY_DATA))

        indexer, _, _ = _make_indexer(tmp_path, tmp_path / ".chroma")
        assert indexer._load_record_pair(sid) is None


class TestIndexAll:
    async def test_indexes_valid_documents(self, tmp_path):
        sid = "entry-001"
        d = tmp_path / sid
        d.mkdir()
        (d / "feedback_summary.json").write_text(json.dumps(SAMPLE_SUMMARY_DATA))
        (d / "feedback_record.json").write_text(json.dumps(SAMPLE_RECORD_DATA))

        indexer, mock_collection, mock_emb = _make_indexer(
            tmp_path, tmp_path / ".chroma"
        )
        count = await indexer.index_all()

        assert count == 1
        mock_emb.embed_texts.assert_called_once()
        mock_collection.add.assert_called_once()

        call_kwargs = mock_collection.add.call_args
        assert call_kwargs.kwargs["ids"] == [sid]
        assert len(call_kwargs.kwargs["documents"]) == 1

    async def test_empty_dir_returns_zero(self, tmp_path):
        indexer, mock_collection, _ = _make_indexer(tmp_path, tmp_path / ".chroma")
        count = await indexer.index_all()
        assert count == 0
        mock_collection.add.assert_not_called()

    async def test_skips_entries_with_no_content(self, tmp_path):
        sid = "entry-no-content"
        d = tmp_path / sid
        d.mkdir()
        empty_summary = {"id": sid, "attributes": {}}
        (d / "feedback_summary.json").write_text(json.dumps(empty_summary))
        (d / "feedback_record.json").write_text(json.dumps(SAMPLE_RECORD_DATA))

        indexer, mock_collection, _ = _make_indexer(tmp_path, tmp_path / ".chroma")
        count = await indexer.index_all()
        assert count == 0
        mock_collection.add.assert_not_called()

    async def test_skips_entries_with_missing_pair(self, tmp_path):
        sid = "entry-no-record"
        d = tmp_path / sid
        d.mkdir()
        (d / "feedback_summary.json").write_text(json.dumps(SAMPLE_SUMMARY_DATA))
        # No feedback_record.json

        indexer, mock_collection, _ = _make_indexer(tmp_path, tmp_path / ".chroma")
        count = await indexer.index_all()
        assert count == 0
        mock_collection.add.assert_not_called()
