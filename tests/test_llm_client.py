"""Tests for src.llm_client — mock AsyncOpenAI."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models import SearchResult


class TestOpenAIClientInit:
    @patch("src.llm_client.AsyncOpenAI")
    def test_init_with_explicit_key(self, mock_cls):
        from src.llm_client import OpenAIClient

        client = OpenAIClient(api_key="sk-test")
        mock_cls.assert_called_once_with(api_key="sk-test")
        assert client.model == "gpt-4o-mini"

    @patch.dict("os.environ", {}, clear=True)
    def test_init_missing_key_raises(self):
        from src.llm_client import OpenAIClient

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIClient()


class TestBuildPrompt:
    def _make_client(self):
        with patch("src.llm_client.AsyncOpenAI"):
            from src.llm_client import OpenAIClient

            return OpenAIClient(api_key="sk-test")

    def test_formats_numbered_feedback(self, sample_search_results):
        client = self._make_client()
        prompt = client._build_prompt("why crashes?", sample_search_results)
        assert "Feedback 1:" in prompt
        assert "Feedback 2:" in prompt
        assert "why crashes?" in prompt

    def test_handles_empty_results(self):
        client = self._make_client()
        prompt = client._build_prompt("query", [])
        assert "query" in prompt
        assert "Feedback 1:" not in prompt


class TestGenerateAnswer:
    async def test_yields_streamed_content_chunks(self):
        with patch("src.llm_client.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            chunk1 = MagicMock()
            chunk1.choices = [MagicMock()]
            chunk1.choices[0].delta.content = "Hello"

            chunk2 = MagicMock()
            chunk2.choices = [MagicMock()]
            chunk2.choices[0].delta.content = " world"

            async def fake_stream():
                yield chunk1
                yield chunk2

            mock_client.chat.completions.create = AsyncMock(
                return_value=fake_stream()
            )

            from src.llm_client import OpenAIClient

            client = OpenAIClient(api_key="sk-test")

            results = [
                SearchResult(
                    content="feedback text",
                    feedback_summary={},
                    feedback_record={},
                    score=0.9,
                )
            ]
            chunks = [c async for c in client.generate_answer("q", results)]
            assert chunks == ["Hello", " world"]

    async def test_skips_falsy_content_chunks(self):
        with patch("src.llm_client.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            chunk_with_content = MagicMock()
            chunk_with_content.choices = [MagicMock()]
            chunk_with_content.choices[0].delta.content = "data"

            chunk_none = MagicMock()
            chunk_none.choices = [MagicMock()]
            chunk_none.choices[0].delta.content = None

            chunk_empty = MagicMock()
            chunk_empty.choices = [MagicMock()]
            chunk_empty.choices[0].delta.content = ""

            async def fake_stream():
                yield chunk_none
                yield chunk_with_content
                yield chunk_empty

            mock_client.chat.completions.create = AsyncMock(
                return_value=fake_stream()
            )

            from src.llm_client import OpenAIClient

            client = OpenAIClient(api_key="sk-test")

            results = [
                SearchResult(
                    content="text",
                    feedback_summary={},
                    feedback_record={},
                    score=0.9,
                )
            ]
            chunks = [c async for c in client.generate_answer("q", results)]
            assert chunks == ["data"]

    async def test_passes_correct_params_to_api(self):
        with patch("src.llm_client.AsyncOpenAI") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client

            async def fake_stream():
                return
                yield  # make it an async generator

            mock_client.chat.completions.create = AsyncMock(
                return_value=fake_stream()
            )

            from src.llm_client import OpenAIClient

            client = OpenAIClient(
                api_key="sk-test", model="gpt-4o", temperature=0.5
            )

            results = [
                SearchResult(
                    content="text",
                    feedback_summary={},
                    feedback_record={},
                    score=0.9,
                )
            ]
            # Consume the generator
            [c async for c in client.generate_answer("my query", results)]

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4o"
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["stream"] is True
            assert len(call_kwargs["messages"]) == 2
