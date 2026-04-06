"""Unit tests cho CaptureOnlyLLMClient."""

from __future__ import annotations

from chahi.infrastructure.llm.capture_only_client import CaptureOnlyLLMClient


class TestCaptureOnlyLLMClient:
    """Tests cho mode dump input không gọi provider."""

    def test_analyze_dumps_file_and_returns_marker(self, tmp_path) -> None:
        client = CaptureOnlyLLMClient(dump_dir=tmp_path)

        result = client.analyze(
            system_prompt="system prompt",
            user_content="user content",
        )

        assert "[CAPTURE_ONLY]" in result
        files = sorted(tmp_path.glob("llm_input_*.md"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "Mode: CAPTURE_ONLY" in content
        assert "system prompt" in content
        assert "user content" in content

    def test_supports_concurrency_false(self, tmp_path) -> None:
        client = CaptureOnlyLLMClient(dump_dir=tmp_path)
        assert client.supports_concurrency is False
