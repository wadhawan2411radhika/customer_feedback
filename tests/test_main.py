"""Tests for main.py — mock all constructors."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


class TestIndexCommand:
    @patch("main.FeedbackIndexer")
    async def test_creates_indexer_and_calls_index_all(self, mock_indexer_cls):
        mock_indexer = AsyncMock()
        mock_indexer.index_all = AsyncMock(return_value=5)
        mock_indexer_cls.return_value = mock_indexer

        from main import index_command

        await index_command(Path("data"), Path(".chroma_db"))

        mock_indexer_cls.assert_called_once_with(
            data_dir=Path("data"), index_path=Path(".chroma_db")
        )
        mock_indexer.index_all.assert_called_once()


class TestQueryCommand:
    @patch("main.RAGPipeline")
    @patch("main.OpenAIClient")
    @patch("main.FeedbackRetriever")
    async def test_creates_pipeline_and_streams(
        self, mock_retriever_cls, mock_llm_cls, mock_pipeline_cls
    ):
        async def fake_query(q):
            yield "answer"

        mock_pipeline = MagicMock()
        mock_pipeline.query = fake_query
        mock_pipeline_cls.return_value = mock_pipeline

        from main import query_command

        await query_command("test query", Path(".chroma_db"), "gpt-4o-mini", 5)

        mock_retriever_cls.assert_called_once_with(
            index_path=Path(".chroma_db"), top_k=5
        )
        mock_llm_cls.assert_called_once_with(model="gpt-4o-mini")
        mock_pipeline_cls.assert_called_once()


class TestMain:
    @patch("main.query_command", new_callable=AsyncMock)
    @patch("main.index_command", new_callable=AsyncMock)
    async def test_dispatches_index_command(self, mock_index, mock_query):
        import sys

        from main import main

        with patch.object(sys, "argv", ["main", "index", "--data-dir", "data"]):
            await main()

        mock_index.assert_called_once()
        mock_query.assert_not_called()

    @patch("main.query_command", new_callable=AsyncMock)
    @patch("main.index_command", new_callable=AsyncMock)
    async def test_dispatches_query_command(self, mock_index, mock_query):
        import sys

        from main import main

        with patch.object(
            sys, "argv", ["main", "query", "what is the feedback?"]
        ):
            await main()

        mock_query.assert_called_once()
        mock_index.assert_not_called()

    @patch("main.query_command", new_callable=AsyncMock)
    @patch("main.index_command", new_callable=AsyncMock)
    async def test_no_command_prints_help(self, mock_index, mock_query, capsys):
        import sys

        from main import main

        with patch.object(sys, "argv", ["main"]):
            await main()

        mock_index.assert_not_called()
        mock_query.assert_not_called()
