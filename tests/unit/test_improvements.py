"""Unit tests cho các cải thiện mới:
- _smart_truncate (word-boundary truncation)
- BeautifulSoup HTML cleaning
- LLM Factory (Strategy Pattern dispatch)
- GeminiClient (mocked)
- LLMSettings provider validation
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest
import requests

from chahi.core.entities import LLMSettings
from chahi.core.interfaces import ILLMClient
from chahi.infrastructure.llm.gemini_client import GeminiClient
from chahi.infrastructure.llm.llm_factory import create_llm_client
from chahi.infrastructure.llm.lm_studio_client import LMStudioClient
from chahi.infrastructure.rss.rss_fetcher import RSSNewsFetcher, _smart_truncate

# ─────────────────────────────────────────────────────────────
# _smart_truncate
# ─────────────────────────────────────────────────────────────


class TestSmartTruncate:
    """Tests cho _smart_truncate() — token-based truncation."""

    def test_short_text_unchanged(self) -> None:
        """Text ngắn hơn token budget phải giữ nguyên."""
        assert _smart_truncate("Ngắn gọn", 500) == "Ngắn gọn"

    def test_exact_length_unchanged(self) -> None:
        """Text vừa đủ budget phải giữ nguyên."""
        text = "hello " * 10  # ~10 tokens
        assert _smart_truncate(text.strip(), 500) == text.strip()

    def test_cuts_when_over_budget(self) -> None:
        """Text vượt token budget phải bị cắt."""
        text = "Lạm phát tăng lên " * 100  # > 100 tokens
        result = _smart_truncate(text, 10)
        assert result.endswith("…")
        assert len(result) < len(text)

    def test_single_long_word_hard_cut(self) -> None:
        """Từ dài hơn budget phải bị cắt."""
        text = "A" * 10000
        result = _smart_truncate(text, 50)
        assert result.endswith("…")
        assert len(result) < len(text)

    def test_empty_string(self) -> None:
        """Chuỗi rỗng phải giữ nguyên."""
        assert _smart_truncate("", 500) == ""


# ─────────────────────────────────────────────────────────────
# BeautifulSoup HTML Cleaning
# ─────────────────────────────────────────────────────────────


class TestBeautifulSoupCleaning:
    """Tests cho _clean_html() dùng BeautifulSoup4."""

    def test_removes_script_tags(self) -> None:
        """Script tags phải bị xóa hoàn toàn (nội dung + tag)."""
        fetcher = RSSNewsFetcher(source_name="Test")
        raw = '<p>Text</p><script>alert("xss")</script><p>More</p>'
        result = fetcher._clean_html(raw)
        assert "alert" not in result
        assert "script" not in result
        assert "Text" in result
        assert "More" in result

    def test_removes_style_tags(self) -> None:
        """Style tags phải bị xóa hoàn toàn."""
        fetcher = RSSNewsFetcher(source_name="Test")
        raw = "<style>.red{color:red}</style><p>Content</p>"
        result = fetcher._clean_html(raw)
        assert "red" not in result
        assert "Content" in result

    def test_nested_complex_html(self) -> None:
        """HTML lồng nhau phức tạp phải xử lý đúng."""
        fetcher = RSSNewsFetcher(source_name="Test")
        raw = """
        <div class="article">
            <h2><a href="/link">Title</a></h2>
            <p>Paragraph <b>bold</b> <i>italic</i></p>
            <ul><li>Item 1</li><li>Item 2</li></ul>
        </div>
        """
        result = fetcher._clean_html(raw)
        assert "Title" in result
        assert "bold" in result
        assert "italic" in result
        assert "Item 1" in result
        assert "<" not in result  # không còn HTML tags

    def test_malformed_html_handled(self) -> None:
        """Malformed HTML không crash."""
        fetcher = RSSNewsFetcher(source_name="Test")
        raw = "<p>Unclosed tag <b>bold <p>new para"
        result = fetcher._clean_html(raw)
        assert "Unclosed" in result
        assert "bold" in result


# ─────────────────────────────────────────────────────────────
# LLMSettings provider validation
# ─────────────────────────────────────────────────────────────


class TestLLMSettingsProvider:
    """Tests cho LLMSettings provider field."""

    def test_default_provider(self) -> None:
        """Default provider phải là lm_studio."""
        settings = LLMSettings()
        assert settings.provider == "lm_studio"

    def test_gemini_provider(self) -> None:
        """Gemini provider phải hợp lệ."""
        settings = LLMSettings(
            provider="gemini",
            api_key="AIzaTest",
            model_name="gemini-2.0-flash",
        )
        assert settings.provider == "gemini"

    def test_invalid_provider_raises(self) -> None:
        """Provider không hợp lệ phải raise ValueError."""
        with pytest.raises(ValueError, match="provider"):
            LLMSettings(provider="claude")

    def test_lm_studio_requires_api_base(self) -> None:
        """lm_studio provider phải có api_base."""
        with pytest.raises(ValueError, match="api_base"):
            LLMSettings(provider="lm_studio", api_base="  ")


# ─────────────────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────────────────


class TestLLMFactory:
    """Tests cho create_llm_client() factory."""

    @patch("chahi.infrastructure.llm.lm_studio_client.openai.OpenAI")
    def test_creates_lm_studio(self, mock_openai: MagicMock) -> None:
        """provider=lm_studio phải trả về LMStudioClient."""
        settings = LLMSettings(provider="lm_studio")
        client = create_llm_client(settings)
        assert isinstance(client, LMStudioClient)

    @patch("chahi.infrastructure.llm.gemini_client.genai.Client")
    def test_creates_gemini(self, mock_genai: MagicMock) -> None:
        """provider=gemini phải trả về GeminiClient."""
        settings = LLMSettings(
            provider="gemini",
            api_key="test-key",
            model_name="gemini-2.0-flash",
        )
        client = create_llm_client(settings)
        assert isinstance(client, GeminiClient)

    def test_invalid_provider_raises(self) -> None:
        """Provider không hỗ trợ phải raise ValueError."""
        # Bypass LLMSettings validation bằng object.__setattr__
        settings = LLMSettings()
        object.__setattr__(settings, "provider", "unknown")
        with pytest.raises(ValueError, match="không hỗ trợ"):
            create_llm_client(settings)

    @patch("chahi.infrastructure.llm.lm_studio_client.openai.OpenAI")
    def test_returns_ilm_client(self, mock_openai: MagicMock) -> None:
        """Factory phải trả về instance của ILLMClient."""
        settings = LLMSettings(provider="lm_studio")
        client = create_llm_client(settings)
        assert isinstance(client, ILLMClient)


# ─────────────────────────────────────────────────────────────
# GeminiClient (mocked)
# ─────────────────────────────────────────────────────────────


class TestGeminiClient:
    """Tests cho GeminiClient — mocked google-genai."""

    @patch("chahi.infrastructure.llm.gemini_client.genai.Client")
    def test_returns_content(self, mock_client_cls: MagicMock) -> None:
        """analyze() phải trả về text từ Gemini response."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "# Báo cáo\nPhân tích..."
        mock_client.models.generate_content.return_value = mock_response

        settings = LLMSettings(
            provider="gemini", api_key="test", model_name="gemini-flash"
        )
        client = GeminiClient(settings=settings)
        result = client.analyze("system", "user content")

        assert result == "# Báo cáo\nPhân tích..."

    @patch("chahi.infrastructure.llm.gemini_client.genai.Client")
    def test_empty_response_raises(self, mock_client_cls: MagicMock) -> None:
        """Response rỗng phải raise RuntimeError."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = ""
        mock_client.models.generate_content.return_value = mock_response

        settings = LLMSettings(
            provider="gemini", api_key="test", model_name="gemini-flash"
        )
        client = GeminiClient(settings=settings)

        with pytest.raises(RuntimeError, match="rỗng"):
            client.analyze("system", "user")

    @patch("chahi.infrastructure.llm.gemini_client.genai.Client")
    def test_auth_error_raises_connection(self, mock_client_cls: MagicMock) -> None:
        """Lỗi xác thực phải raise ConnectionError."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.models.generate_content.side_effect = Exception(
            "Invalid API_KEY provided"
        )

        settings = LLMSettings(
            provider="gemini", api_key="bad-key", model_name="gemini-flash"
        )
        client = GeminiClient(settings=settings)

        with pytest.raises(ConnectionError, match="API key"):
            client.analyze("system", "user")


# ─────────────────────────────────────────────────────────────
# MemorySettings validation
# ─────────────────────────────────────────────────────────────


class TestMemorySettings:
    """Tests cho MemorySettings entity."""

    def test_default_type_file(self) -> None:
        """Default type phải là 'file'."""
        from chahi.core.entities import MemorySettings

        settings = MemorySettings()
        assert settings.type == "file"

    def test_mcp_type_valid(self) -> None:
        """MCP type với URL phải hợp lệ."""
        from chahi.core.entities import MemorySettings

        settings = MemorySettings(type="mcp", url="http://localhost:8080")
        assert settings.type == "mcp"
        assert settings.url == "http://localhost:8080"

    def test_invalid_type_raises(self) -> None:
        """Type không hợp lệ phải raise ValueError."""
        from chahi.core.entities import MemorySettings

        with pytest.raises(ValueError, match="type"):
            MemorySettings(type="redis")

    def test_mcp_without_url_raises(self) -> None:
        """MCP type không có URL phải raise ValueError."""
        from chahi.core.entities import MemorySettings

        with pytest.raises(ValueError, match="url"):
            MemorySettings(type="mcp", url="")


# ─────────────────────────────────────────────────────────────
# Memory Factory
# ─────────────────────────────────────────────────────────────


class TestMemoryFactory:
    """Tests cho create_memory_manager() factory."""

    def test_creates_file_manager(self) -> None:
        """type=file phải trả về FileMemoryManager."""
        from chahi.core.entities import MemorySettings
        from chahi.infrastructure.memory.file_memory_manager import FileMemoryManager
        from chahi.infrastructure.memory.memory_factory import create_memory_manager

        settings = MemorySettings(type="file")
        mgr = create_memory_manager(settings)
        assert isinstance(mgr, FileMemoryManager)

    def test_creates_mcp_manager(self) -> None:
        """type=mcp phải trả về MCPHttpMemoryManager."""
        from chahi.core.entities import MemorySettings
        from chahi.infrastructure.memory.mcp_memory_manager import MCPHttpMemoryManager
        from chahi.infrastructure.memory.memory_factory import create_memory_manager

        settings = MemorySettings(type="mcp", url="http://localhost:8080")
        mgr = create_memory_manager(settings)
        assert isinstance(mgr, MCPHttpMemoryManager)

    def test_invalid_type_raises(self) -> None:
        """Type không hợp lệ phải raise ValueError."""
        from chahi.core.entities import MemorySettings
        from chahi.infrastructure.memory.memory_factory import create_memory_manager

        settings = MemorySettings(type="file")
        object.__setattr__(settings, "type", "unknown")
        with pytest.raises(ValueError, match="không hỗ trợ"):
            create_memory_manager(settings)


# ─────────────────────────────────────────────────────────────
# FileMemoryManager File Locking
# ─────────────────────────────────────────────────────────────


class TestFileMemoryManagerLocking:
    """Tests cho FileLock trong FileMemoryManager."""

    def test_creates_lock_file(self, tmp_path: Path) -> None:
        """FileMemoryManager phải tạo FileLock instance."""
        from filelock import FileLock

        from chahi.infrastructure.memory.file_memory_manager import (
            FileMemoryManager,
        )

        mgr = FileMemoryManager(memory_dir=tmp_path)
        mgr.save_context("Test data")

        # Verify lock exists and is a FileLock
        assert hasattr(mgr, "_lock")
        assert isinstance(mgr._lock, FileLock)
        # Verify data was saved correctly
        assert (tmp_path / "last_context.md").exists()

    def test_save_and_retrieve_under_lock(self, tmp_path: Path) -> None:
        """Save → Retrieve phải trả về đúng dữ liệu dưới lock."""
        from chahi.infrastructure.memory.file_memory_manager import (
            FileMemoryManager,
        )

        mgr = FileMemoryManager(memory_dir=tmp_path)
        mgr.save_context("Nhận định test")

        result = mgr.retrieve_last_context()
        assert result is not None
        assert "Nhận định test" in result

    def test_lock_timeout_raises(self, tmp_path: Path) -> None:
        """Lock bị giữ bởi process khác → RuntimeError sau timeout."""
        from filelock import FileLock

        from chahi.infrastructure.memory.file_memory_manager import (
            FileMemoryManager,
        )

        mgr = FileMemoryManager(memory_dir=tmp_path)
        # Pre-create file to avoid early return
        (tmp_path / "last_context.md").write_text("old data")

        # Hold the lock externally
        lock_path = str(tmp_path / "last_context.md.lock")
        external_lock = FileLock(lock_path, timeout=0)
        external_lock.acquire()

        try:
            # Override lock timeout to 0 for fast test
            mgr._lock = FileLock(lock_path, timeout=0)
            with pytest.raises(RuntimeError, match="lock"):
                mgr.retrieve_last_context()
        finally:
            external_lock.release()


# ─────────────────────────────────────────────────────────────
# MCPHttpMemoryManager (mocked HTTP)
# ─────────────────────────────────────────────────────────────


class TestMCPHttpMemoryManager:
    """Tests cho MCPHttpMemoryManager — mocked requests."""

    def test_retrieve_returns_text(self) -> None:
        """retrieve phải trả về text từ MCP response."""
        from chahi.core.entities import MemorySettings
        from chahi.infrastructure.memory.mcp_memory_manager import MCPHttpMemoryManager

        settings = MemorySettings(type="mcp", url="http://localhost:8080")
        mgr = MCPHttpMemoryManager(settings=settings)

        with patch.object(mgr, "_call_tool") as mock_call:
            mock_call.return_value = {"results": [{"text": "Dầu tăng 3%, vàng giảm"}]}
            result = mgr.retrieve_last_context()
            assert result == "Dầu tăng 3%, vàng giảm"

    def test_retrieve_returns_none_on_empty(self) -> None:
        """retrieve trả None khi MCP không có data."""
        from chahi.core.entities import MemorySettings
        from chahi.infrastructure.memory.mcp_memory_manager import MCPHttpMemoryManager

        settings = MemorySettings(type="mcp", url="http://localhost:8080")
        mgr = MCPHttpMemoryManager(settings=settings)

        with patch.object(mgr, "_call_tool") as mock_call:
            mock_call.return_value = None
            result = mgr.retrieve_last_context()
            assert result is None

    def test_save_calls_store_tool(self) -> None:
        """save phải gọi store_working_context tool."""
        from chahi.core.entities import MemorySettings
        from chahi.infrastructure.memory.mcp_memory_manager import MCPHttpMemoryManager

        settings = MemorySettings(type="mcp", url="http://localhost:8080")
        mgr = MCPHttpMemoryManager(settings=settings)

        with patch.object(mgr, "_call_tool") as mock_call:
            mock_call.return_value = {"status": "ok"}
            mgr.save_context("Test summary")
            mock_call.assert_called_once()
            args = mock_call.call_args[1]
            assert args["tool_name"] == "store_working_context"
            assert "Test summary" in args["arguments"]["text_data"]
            assert "[ChaHi Report" in args["arguments"]["text_data"]

    def test_connection_error_graceful(self) -> None:
        """Lỗi kết nối MCP phải trả None, không crash."""
        from chahi.core.entities import MemorySettings
        from chahi.infrastructure.memory.mcp_memory_manager import MCPHttpMemoryManager

        settings = MemorySettings(type="mcp", url="http://localhost:9999")
        mgr = MCPHttpMemoryManager(settings=settings)

        with patch.object(mgr._session, "post") as mock_post:
            mock_post.side_effect = requests.ConnectionError("refused")
            result = mgr.retrieve_last_context()
            assert result is None  # graceful, không crash

    def test_timeout_graceful(self) -> None:
        """Timeout MCP phải trả None, không crash."""
        from chahi.core.entities import MemorySettings
        from chahi.infrastructure.memory.mcp_memory_manager import MCPHttpMemoryManager

        settings = MemorySettings(type="mcp", url="http://localhost:9999")
        mgr = MCPHttpMemoryManager(settings=settings)

        with patch.object(mgr._session, "post") as mock_post:
            mock_post.side_effect = requests.Timeout("timed out")
            result = mgr.retrieve_last_context()
            assert result is None

    def test_extract_content_pattern(self) -> None:
        """_extract_text_from_result xử lý nhiều patterns."""
        from chahi.core.entities import MemorySettings
        from chahi.infrastructure.memory.mcp_memory_manager import MCPHttpMemoryManager

        settings = MemorySettings(type="mcp", url="http://test")
        mgr = MCPHttpMemoryManager(settings=settings)

        # Pattern 1: content[0].text
        result = mgr._extract_text_from_result({"content": [{"text": "abc"}]})
        assert result == "abc"

        # Pattern 2: results[0].document
        result = mgr._extract_text_from_result({"results": [{"document": "xyz"}]})
        assert result == "xyz"

        # Pattern 3: output
        result = mgr._extract_text_from_result({"output": "hello"})
        assert result == "hello"

        # Empty
        result = mgr._extract_text_from_result({})
        assert result is None


# ─────────────────────────────────────────────────────────────
# Dual-Protocol MCP (rest vs jsonrpc)
# ─────────────────────────────────────────────────────────────


class TestMCPProtocolSelection:
    """Tests cho dual-protocol MCP Memory Manager."""

    def test_protocol_default_rest(self) -> None:
        """Default protocol phải là 'rest'."""
        from chahi.core.entities import MemorySettings

        settings = MemorySettings(type="mcp", url="http://localhost:8080")
        assert settings.protocol == "rest"

    def test_protocol_jsonrpc_valid(self) -> None:
        """jsonrpc protocol phải hợp lệ."""
        from chahi.core.entities import MemorySettings

        settings = MemorySettings(
            type="mcp", url="http://localhost:8080", protocol="jsonrpc"
        )
        assert settings.protocol == "jsonrpc"

    def test_protocol_invalid_raises(self) -> None:
        """Protocol không hợp lệ phải raise ValueError."""
        from chahi.core.entities import MemorySettings

        with pytest.raises(ValueError, match="protocol"):
            MemorySettings(type="mcp", url="http://localhost:8080", protocol="grpc")

    def test_rest_calls_call_tool_endpoint(self) -> None:
        """protocol=rest phải gọi POST /call-tool."""
        from chahi.core.entities import MemorySettings
        from chahi.infrastructure.memory.mcp_memory_manager import MCPHttpMemoryManager

        settings = MemorySettings(type="mcp", url="http://localhost:8080")
        mgr = MCPHttpMemoryManager(settings=settings)

        with patch.object(mgr, "_http_post") as mock_post:
            mock_post.return_value = {"results": [{"text": "data"}]}
            mgr._call_tool("search_memory", {"query": "test"})
            mock_post.assert_called_once()
            url_arg = mock_post.call_args[0][0]
            assert url_arg == "http://localhost:8080/call-tool"

    def test_jsonrpc_calls_message_endpoint(self) -> None:
        """protocol=jsonrpc phải gọi SSE handshake + POST /message."""
        from chahi.core.entities import MemorySettings
        from chahi.infrastructure.memory.mcp_memory_manager import MCPHttpMemoryManager

        settings = MemorySettings(
            type="mcp", url="http://localhost:8080", protocol="jsonrpc"
        )
        mgr = MCPHttpMemoryManager(settings=settings)

        with patch.object(mgr, "_call_tool_jsonrpc") as mock_jsonrpc:
            mock_jsonrpc.return_value = {"result": {"content": [{"text": "ok"}]}}
            mgr._call_tool("search_memory", {"query": "test"})
            mock_jsonrpc.assert_called_once_with("search_memory", {"query": "test"})

    def test_jsonrpc_sse_failure_returns_none(self) -> None:
        """SSE handshake thất bại → _call_tool trả None."""
        from chahi.core.entities import MemorySettings
        from chahi.infrastructure.memory.mcp_memory_manager import MCPHttpMemoryManager

        settings = MemorySettings(
            type="mcp", url="http://localhost:8080", protocol="jsonrpc"
        )
        mgr = MCPHttpMemoryManager(settings=settings)

        with patch.object(mgr._session, "get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("refused")
            result = mgr._call_tool("search_memory", {"query": "test"})
            assert result is None
