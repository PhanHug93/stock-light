"""Unit tests cho OpenAIClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import openai
import pytest

from chahi.core.entities import LLMSettings
from chahi.infrastructure.llm.openai_client import OpenAIClient


@pytest.fixture()
def settings() -> LLMSettings:
    """LLMSettings mẫu cho OpenAI."""
    return LLMSettings(
        provider="openai",
        api_base="https://api.openai.com/v1",
        api_key="sk-test",
        model_name="gpt-4o-mini",
        temperature=0.2,
    )


def _make_mock_response(content: str | None, finish_reason: str = "stop") -> MagicMock:
    """Tạo mock response giống openai ChatCompletion."""
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    return mock_response


class TestOpenAIClientAnalyze:
    """Tests cho OpenAIClient.analyze()."""

    @patch("chahi.infrastructure.llm.openai_client.openai.OpenAI")
    def test_constructor_passes_settings(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """Constructor phải truyền đúng base_url, api_key, timeout."""
        OpenAIClient(settings=settings)

        mock_openai_cls.assert_called_once_with(
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            timeout=120,
        )

    @patch("chahi.infrastructure.llm.openai_client.openai.OpenAI")
    def test_empty_api_key_uses_env_fallback(self, mock_openai_cls: MagicMock) -> None:
        """api_key rỗng phải để OpenAI SDK fallback sang env var."""
        empty_key_settings = LLMSettings(
            provider="openai",
            api_base="https://api.openai.com/v1",
            api_key="   ",
            model_name="gpt-4o-mini",
        )
        OpenAIClient(settings=empty_key_settings)

        mock_openai_cls.assert_called_once_with(
            api_key=None,
            base_url="https://api.openai.com/v1",
            timeout=120,
        )

    @patch("chahi.infrastructure.llm.openai_client.openai.OpenAI")
    def test_returns_content(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """analyze() phải trả về content hợp lệ."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content="  # Report\nNội dung  "
        )

        client = OpenAIClient(settings=settings)
        result = client.analyze(system_prompt="sys", user_content="user")

        assert result == "# Report\nNội dung"
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
            ],
            temperature=0.2,
        )


class TestOpenAIClientErrors:
    """Tests cho xử lý lỗi OpenAIClient."""

    @patch("chahi.infrastructure.llm.openai_client.openai.OpenAI")
    def test_auth_error_raises_connection(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """AuthenticationError phải raise ConnectionError."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "invalid api key"}}
        mock_client.chat.completions.create.side_effect = openai.AuthenticationError(
            message="invalid api key",
            response=mock_response,
            body={"error": {"message": "invalid api key"}},
        )

        client = OpenAIClient(settings=settings)
        with pytest.raises(ConnectionError, match="OPENAI_API_KEY"):
            client.analyze(system_prompt="s", user_content="u")

    @patch("chahi.infrastructure.llm.openai_client.openai.OpenAI")
    def test_connection_error_raises_connection(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """APIConnectionError phải raise ConnectionError."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
            request=MagicMock()
        )

        client = OpenAIClient(settings=settings)
        with pytest.raises(ConnectionError, match="kết nối"):
            client.analyze(system_prompt="s", user_content="u")

    @patch("chahi.infrastructure.llm.openai_client.openai.OpenAI")
    def test_empty_choices_raises_runtime(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """Response không có choices phải raise RuntimeError."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.chat.completions.create.return_value = mock_response

        client = OpenAIClient(settings=settings)
        with pytest.raises(RuntimeError, match="choices"):
            client.analyze(system_prompt="s", user_content="u")

    @patch("chahi.infrastructure.llm.openai_client.openai.OpenAI")
    def test_empty_content_raises_runtime(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """Response content rỗng phải raise RuntimeError."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content="   "
        )

        client = OpenAIClient(settings=settings)
        with pytest.raises(RuntimeError, match="rỗng"):
            client.analyze(system_prompt="s", user_content="u")
