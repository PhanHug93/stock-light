"""Unit tests cho RecordingLLMClient."""

from __future__ import annotations

from unittest.mock import MagicMock

from chahi.infrastructure.llm.recording_client import RecordingLLMClient


class TestRecordingLLMClient:
    """Tests cho wrapper dump input trước khi gọi provider."""

    def test_dump_file_created_with_expected_content(self, tmp_path) -> None:
        delegate = MagicMock()
        delegate.supports_concurrency = True
        delegate.analyze.return_value = "ok"

        client = RecordingLLMClient(delegate=delegate, dump_dir=tmp_path)
        result = client.analyze(
            system_prompt="Bạn là analyst",
            user_content="Tin tức A, B, C",
        )

        assert result == "ok"
        delegate.analyze.assert_called_once_with(
            system_prompt="Bạn là analyst",
            user_content="Tin tức A, B, C",
        )

        files = sorted(tmp_path.glob("llm_input_*.md"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "## System Prompt" in content
        assert "Bạn là analyst" in content
        assert "## User Content" in content
        assert "Tin tức A, B, C" in content

    def test_supports_concurrency_delegated(self, tmp_path) -> None:
        delegate = MagicMock()
        delegate.supports_concurrency = False
        delegate.analyze.return_value = "ok"

        client = RecordingLLMClient(delegate=delegate, dump_dir=tmp_path)
        assert client.supports_concurrency is False

    def test_file_counter_increments(self, tmp_path) -> None:
        delegate = MagicMock()
        delegate.supports_concurrency = True
        delegate.analyze.return_value = "ok"

        client = RecordingLLMClient(delegate=delegate, dump_dir=tmp_path)
        client.analyze(system_prompt="s1", user_content="u1")
        client.analyze(system_prompt="s2", user_content="u2")

        names = sorted(f.name for f in tmp_path.glob("llm_input_*.md"))
        assert len(names) == 2
        assert names[0].startswith("llm_input_001_")
        assert names[1].startswith("llm_input_002_")
