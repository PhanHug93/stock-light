"""Unit tests cho LMStudioClient.

Sử dụng unittest.mock để mock openai.OpenAI client,
không cần LM Studio thực sự chạy khi test.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import openai
import pytest

from chahi.core.entities import LLMSettings
from chahi.infrastructure.llm.lm_studio_client import LMStudioClient


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────


@pytest.fixture()
def settings() -> LLMSettings:
    """LLMSettings mẫu cho test."""
    return LLMSettings(
        api_base="http://localhost:1234/v1",
        api_key="lm-studio",
        model_name="test-model",
        temperature=0.1,
    )


def _make_mock_response(content: str | None, finish_reason: str = "stop") -> MagicMock:
    """Tạo mock response giống openai ChatCompletion.

    Args:
        content: Nội dung message.content. None nếu muốn test empty.
        finish_reason: Lý do kết thúc (stop, length, ...).

    Returns:
        MagicMock giống cấu trúc ChatCompletion response.
    """
    mock_message = MagicMock()
    mock_message.content = content

    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    return mock_response


# ─────────────────────────────────────────────────────────────
# Happy Path
# ─────────────────────────────────────────────────────────────


class TestLMStudioClientAnalyze:
    """Tests cho analyze() — happy path."""

    @patch("chahi.infrastructure.llm.lm_studio_client.openai.OpenAI")
    def test_returns_content(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """analyze() phải trả về content từ LLM response."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content="# Báo cáo\nNội dung phân tích..."
        )

        client = LMStudioClient(settings=settings)
        result = client.analyze(
            system_prompt="Bạn là chuyên gia tài chính.",
            user_content="Tin tức: Fed giữ lãi suất.",
        )

        assert result == "# Báo cáo\nNội dung phân tích..."

    @patch("chahi.infrastructure.llm.lm_studio_client.openai.OpenAI")
    def test_correct_api_call(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """analyze() phải gọi API với đúng model, messages, temperature."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content="response"
        )

        client = LMStudioClient(settings=settings)
        client.analyze(
            system_prompt="system",
            user_content="user",
        )

        mock_client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[
                {"role": "system", "content": "system"},
                {"role": "user", "content": "user"},
            ],
            temperature=0.1,
        )

    @patch("chahi.infrastructure.llm.lm_studio_client.openai.OpenAI")
    def test_strips_whitespace(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """analyze() phải strip whitespace thừa từ response."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content="\n  Nội dung  \n"
        )

        client = LMStudioClient(settings=settings)
        result = client.analyze(system_prompt="s", user_content="u")

        assert result == "Nội dung"

    @patch("chahi.infrastructure.llm.lm_studio_client.openai.OpenAI")
    def test_constructor_passes_settings_with_timeout(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """Constructor phải truyền đúng base_url, api_key, và timeout tới OpenAI client."""
        LMStudioClient(settings=settings)

        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            timeout=120,
        )

    @patch("chahi.infrastructure.llm.lm_studio_client.openai.OpenAI")
    def test_custom_timeout(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """Constructor phải hỗ trợ custom timeout."""
        LMStudioClient(settings=settings, timeout=30)

        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio",
            timeout=30,
        )


# ─────────────────────────────────────────────────────────────
# Error Cases
# ─────────────────────────────────────────────────────────────


class TestLMStudioClientErrors:
    """Tests cho analyze() — error handling."""

    @patch("chahi.infrastructure.llm.lm_studio_client.openai.OpenAI")
    def test_connection_error_message(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """APIConnectionError phải raise ConnectionError với thông báo tiếng Việt."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = (
            openai.APIConnectionError(request=MagicMock())
        )

        client = LMStudioClient(settings=settings)

        with pytest.raises(ConnectionError, match="Start Server"):
            client.analyze(system_prompt="s", user_content="u")

    @patch("chahi.infrastructure.llm.lm_studio_client.openai.OpenAI")
    def test_empty_choices_raises(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """Response không có choices phải raise RuntimeError."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.chat.completions.create.return_value = mock_response

        client = LMStudioClient(settings=settings)

        with pytest.raises(RuntimeError, match="choices"):
            client.analyze(system_prompt="s", user_content="u")

    @patch("chahi.infrastructure.llm.lm_studio_client.openai.OpenAI")
    def test_empty_content_raises(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """Response với content rỗng phải raise RuntimeError."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content="   "
        )

        client = LMStudioClient(settings=settings)

        with pytest.raises(RuntimeError, match="rỗng"):
            client.analyze(system_prompt="s", user_content="u")

    @patch("chahi.infrastructure.llm.lm_studio_client.openai.OpenAI")
    def test_none_content_raises(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """Response với content=None phải raise RuntimeError."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            content=None
        )

        client = LMStudioClient(settings=settings)

        with pytest.raises(RuntimeError, match="rỗng"):
            client.analyze(system_prompt="s", user_content="u")

    @patch("chahi.infrastructure.llm.lm_studio_client.openai.OpenAI")
    def test_timeout_error(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """APITimeoutError phải raise RuntimeError với message timeout."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = (
            openai.APITimeoutError(request=MagicMock())
        )

        client = LMStudioClient(settings=settings)

        with pytest.raises(RuntimeError, match="timeout"):
            client.analyze(system_prompt="s", user_content="u")

    @patch("chahi.infrastructure.llm.lm_studio_client.openai.OpenAI")
    def test_rate_limit_error(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """RateLimitError phải raise RuntimeError với message quá tải."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"error": {"message": "rate limited"}}
        mock_client.chat.completions.create.side_effect = (
            openai.RateLimitError(
                message="rate limited",
                response=mock_response,
                body={"error": {"message": "rate limited"}},
            )
        )

        client = LMStudioClient(settings=settings)

        with pytest.raises(RuntimeError, match="quá tải"):
            client.analyze(system_prompt="s", user_content="u")

    @patch("chahi.infrastructure.llm.lm_studio_client.openai.OpenAI")
    def test_api_status_error(
        self, mock_openai_cls: MagicMock, settings: LLMSettings
    ) -> None:
        """APIStatusError phải raise RuntimeError với HTTP status code."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": {"message": "internal error"}}
        mock_client.chat.completions.create.side_effect = (
            openai.InternalServerError(
                message="internal error",
                response=mock_response,
                body={"error": {"message": "internal error"}},
            )
        )

        client = LMStudioClient(settings=settings)

        with pytest.raises(RuntimeError, match="HTTP"):
            client.analyze(system_prompt="s", user_content="u")
