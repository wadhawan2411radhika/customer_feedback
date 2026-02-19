"""Tests for src.embeddings — mock AsyncOpenAI."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.conftest import SAMPLE_EMBEDDING


class TestOpenAIEmbeddingsInit:
    @patch("src.embeddings.AsyncOpenAI")
    def test_init_with_explicit_key(self, mock_cls):
        from src.embeddings import OpenAIEmbeddings

        emb = OpenAIEmbeddings(api_key="sk-test")
        mock_cls.assert_called_once_with(api_key="sk-test")
        assert emb.model == "text-embedding-3-small"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env"})
    @patch("src.embeddings.AsyncOpenAI")
    def test_init_env_var_fallback(self, mock_cls):
        from src.embeddings import OpenAIEmbeddings

        OpenAIEmbeddings()
        mock_cls.assert_called_once_with(api_key="sk-env")

    @patch.dict("os.environ", {}, clear=True)
    def test_init_missing_key_raises(self):
        from src.embeddings import OpenAIEmbeddings

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIEmbeddings()


class TestEmbedBatch:
    @patch("src.embeddings.AsyncOpenAI")
    async def test_embed_batch_returns_embeddings(self, mock_cls):
        from src.embeddings import OpenAIEmbeddings

        emb_obj = MagicMock()
        emb_obj.embedding = SAMPLE_EMBEDDING
        response = MagicMock()
        response.data = [emb_obj]

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=response)
        mock_cls.return_value = mock_client

        emb = OpenAIEmbeddings(api_key="sk-test")
        result = await emb._embed_batch(["hello"])

        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input=["hello"]
        )
        assert result == [SAMPLE_EMBEDDING]


class TestEmbedTexts:
    @patch("src.embeddings.AsyncOpenAI")
    async def test_empty_input_returns_empty(self, mock_cls):
        from src.embeddings import OpenAIEmbeddings

        emb = OpenAIEmbeddings(api_key="sk-test")
        result = await emb.embed_texts([])
        assert result == []

    @patch("src.embeddings.AsyncOpenAI")
    async def test_single_batch(self, mock_cls):
        from src.embeddings import OpenAIEmbeddings

        emb_obj = MagicMock()
        emb_obj.embedding = [0.1, 0.2]
        response = MagicMock()
        response.data = [emb_obj]

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=response)
        mock_cls.return_value = mock_client

        emb = OpenAIEmbeddings(api_key="sk-test")
        result = await emb.embed_texts(["text1"])

        assert result == [[0.1, 0.2]]

    @patch("src.embeddings.AsyncOpenAI")
    async def test_multiple_batches_preserves_order(self, mock_cls):
        from src.embeddings import OpenAIEmbeddings

        def make_response(embeddings):
            objs = []
            for e in embeddings:
                obj = MagicMock()
                obj.embedding = e
                objs.append(obj)
            resp = MagicMock()
            resp.data = objs
            return resp

        call_count = 0

        async def fake_create(**kwargs):
            nonlocal call_count
            batch = kwargs["input"]
            call_count += 1
            return make_response([[float(call_count)] * len(batch)])

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(side_effect=fake_create)
        mock_cls.return_value = mock_client

        emb = OpenAIEmbeddings(api_key="sk-test", batch_size=2)
        result = await emb.embed_texts(["a", "b", "c"])

        assert len(result) == 2  # 2 batches: [a,b] and [c]


class TestEmbedQuery:
    @patch("src.embeddings.AsyncOpenAI")
    async def test_embed_query_returns_single_embedding(self, mock_cls):
        from src.embeddings import OpenAIEmbeddings

        emb_obj = MagicMock()
        emb_obj.embedding = SAMPLE_EMBEDDING

        response = MagicMock()
        response.data = [emb_obj]

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=response)
        mock_cls.return_value = mock_client

        emb = OpenAIEmbeddings(api_key="sk-test")
        result = await emb.embed_query("what is this?")

        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input=["what is this?"]
        )
        assert result == SAMPLE_EMBEDDING
