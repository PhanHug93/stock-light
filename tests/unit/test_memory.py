"""Unit tests cho Phase 6: Memory Integration.

Tests cho:
- FileMemoryManager (read/write file)
- Feedback loop (use case with memory)
- _extract_summary helper
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from chahi.core.entities import SourceCategory, SourceConfig
from chahi.core.services.report_parser import extract_summary
from chahi.core.use_cases import GenerateMacroReportUseCase
from chahi.infrastructure.memory.file_memory_manager import FileMemoryManager

if TYPE_CHECKING:
    from pathlib import Path

# ─────────────────────────────────────────────────────────────
# FileMemoryManager
# ─────────────────────────────────────────────────────────────


class TestFileMemoryManager:
    """Tests cho FileMemoryManager."""

    def test_retrieve_returns_none_when_no_file(self, tmp_path: Path) -> None:
        """Lần đầu chạy chưa có file → None."""
        mgr = FileMemoryManager(memory_dir=tmp_path / "empty")
        assert mgr.retrieve_last_context() is None

    def test_save_and_retrieve(self, tmp_path: Path) -> None:
        """Save rồi retrieve phải trả về nội dung đã lưu."""
        mgr = FileMemoryManager(memory_dir=tmp_path)
        mgr.save_context("Nhận định: Dầu tăng, Vàng giảm.")

        result = mgr.retrieve_last_context()
        assert result is not None
        assert "Nhận định: Dầu tăng, Vàng giảm." in result

    def test_save_overwrites(self, tmp_path: Path) -> None:
        """Lưu lần 2 phải ghi đè nội dung cũ."""
        mgr = FileMemoryManager(memory_dir=tmp_path)
        mgr.save_context("Phiên 1: Dầu tăng")
        mgr.save_context("Phiên 2: Dầu giảm")

        result = mgr.retrieve_last_context()
        assert result is not None
        assert "Phiên 2" in result
        assert "Phiên 1" not in result

    def test_save_creates_directory(self, tmp_path: Path) -> None:
        """Lưu phải tự tạo thư mục nếu chưa tồn tại."""
        deep_dir = tmp_path / "deep" / "nested" / "memory"
        mgr = FileMemoryManager(memory_dir=deep_dir)
        mgr.save_context("Test content")

        assert deep_dir.exists()
        assert mgr.retrieve_last_context() is not None

    def test_saved_file_contains_timestamp(self, tmp_path: Path) -> None:
        """File lưu phải có timestamp header."""
        mgr = FileMemoryManager(memory_dir=tmp_path)
        mgr.save_context("Content")

        file_content = (tmp_path / "last_context.md").read_text()
        assert "<!-- [default] Saved:" in file_content
        assert "UTC -->" in file_content


# ─────────────────────────────────────────────────────────────
# _extract_summary
# ─────────────────────────────────────────────────────────────


class TestExtractSummary:
    """Tests cho _extract_summary() helper."""

    def test_extracts_summary_section(self) -> None:
        """Phải trích xuất nội dung sau ## Tổng kết."""
        report = """# Báo cáo
## 1. Dầu
Nội dung...
## 6. 📋 Tổng kết
- Dầu tăng 3%
- Vàng giảm nhẹ
- Crypto sideway
"""
        result = extract_summary(report)
        assert "Dầu tăng 3%" in result
        assert "Vàng giảm nhẹ" in result

    def test_fallback_when_no_summary(self) -> None:
        """Không có section Tổng kết → lấy 500 chars cuối."""
        report = "Nội dung ngắn gọn không có section tổng kết."
        result = extract_summary(report)
        assert result == report.strip()

    def test_fallback_long_report(self) -> None:
        """Report dài không có Tổng kết → lấy 500 chars cuối."""
        report = "A" * 1000
        result = extract_summary(report)
        assert len(result) == 500

    def test_matches_without_emoji(self) -> None:
        """Regex phải match khi LLM quên emoji 📋."""
        report = "## 6. Tổng kết\n- Dầu tăng"
        result = extract_summary(report)
        assert "Dầu tăng" in result

    def test_matches_different_numbering(self) -> None:
        """Regex phải match với format 'Phần 6:' hoặc '6.'."""
        report = "## Tổng Kết\n- Vàng giảm nhẹ"
        result = extract_summary(report)
        assert "Vàng giảm" in result

    def test_matches_uppercase(self) -> None:
        """Regex phải match 'TỔNG KẾT' (nhưng giữ case-sensitive vì có [Tt][Kk])."""
        report = "## 6. 📋 Tổng kết — Nhận định\n- Crypto sideway"
        result = extract_summary(report)
        assert "Crypto sideway" in result

    def test_matches_no_number(self) -> None:
        """Regex phải match khi không có số thứ tự."""
        report = "## 📋 Tổng kết\n- BTC giảm 5%"
        result = extract_summary(report)
        assert "BTC giảm 5%" in result


# ─────────────────────────────────────────────────────────────
# Feedback Loop integration
# ─────────────────────────────────────────────────────────────


def _make_source(
    name: str = "Test", url: str = "https://example.com/rss"
) -> SourceConfig:
    return SourceConfig(name=name, url=url, type="rss")


class TestFeedbackLoop:
    """Tests cho memory feedback loop trong execute()."""

    def test_retrieves_and_contextualizes(self) -> None:
        """execute() phải gọi retrieve và chèn context cũ vào user_content."""
        mock_config = MagicMock()
        mock_config.get_sources.return_value = {
            SourceCategory.OIL_MACRO: [_make_source()],
            SourceCategory.GOLD: [],
            SourceCategory.CRYPTO: [],
        }
        mock_fetcher = MagicMock()
        mock_fetcher.supports_batch = False
        mock_fetcher.fetch_news.return_value = [
            MagicMock(
                title="Test",
                summary="Fed giữ lãi suất, vàng tăng nhẹ và Bitcoin ETF hút vốn.",
                source_name="Src",
                published_date=MagicMock(
                    strftime=MagicMock(return_value="01/01/2026 10:00")
                ),
                url="https://example.com",
            )
        ]
        mock_llm = MagicMock()
        mock_llm.analyze.return_value = "## 6. 📋 Tổng kết\n- Kết luận"
        mock_memory = MagicMock()
        mock_memory.retrieve_last_context.return_value = "Hôm qua: Dầu tăng mạnh."
        mock_memory.retrieve_related_context.return_value = None

        use_case = GenerateMacroReportUseCase(
            config_reader=mock_config,
            news_fetcher=mock_fetcher,
            llm_client=mock_llm,
            memory_manager=mock_memory,
        )
        use_case.execute()

        # Kiểm tra retrieve được gọi
        mock_memory.retrieve_last_context.assert_called_once()

        # Kiểm tra context cũ xuất hiện trong user_content
        actual_content = mock_llm.analyze.call_args[1]["user_content"]
        assert "<previous_lessons>" in actual_content
        assert "Hôm qua: Dầu tăng mạnh" in actual_content

    def test_stores_summary_after_report(self) -> None:
        """execute() phải gọi save_context sau khi có report."""
        mock_config = MagicMock()
        mock_config.get_sources.return_value = {
            SourceCategory.OIL_MACRO: [_make_source()],
        }
        mock_fetcher = MagicMock()
        mock_fetcher.supports_batch = False
        mock_fetcher.fetch_news.return_value = [
            MagicMock(
                title="Test",
                summary="Fed giữ lãi suất, vàng tăng nhẹ và Bitcoin ETF hút vốn.",
                source_name="Src",
                published_date=MagicMock(
                    strftime=MagicMock(return_value="01/01/2026 10:00")
                ),
                url="https://example.com",
            )
        ]
        mock_llm = MagicMock()
        mock_llm.analyze.return_value = "## 6. 📋 Tổng kết\n- Dầu giảm 2%"
        mock_memory = MagicMock()
        mock_memory.retrieve_last_context.return_value = None
        mock_memory.retrieve_related_context.return_value = None

        use_case = GenerateMacroReportUseCase(
            config_reader=mock_config,
            news_fetcher=mock_fetcher,
            llm_client=mock_llm,
            memory_manager=mock_memory,
        )
        use_case.execute()

        # Kiểm tra save được gọi với phần tổng kết
        mock_memory.save_context.assert_called_once()
        saved = mock_memory.save_context.call_args[0][0]
        assert "Dầu giảm 2%" in saved

    def test_no_memory_still_works(self) -> None:
        """Không có memory_manager, pipeline vẫn chạy bình thường."""
        mock_config = MagicMock()
        mock_config.get_sources.return_value = {
            SourceCategory.OIL_MACRO: [_make_source()],
        }
        mock_fetcher = MagicMock()
        mock_fetcher.supports_batch = False
        mock_fetcher.fetch_news.return_value = [
            MagicMock(
                title="Test",
                summary="Fed giữ lãi suất, vàng tăng nhẹ và Bitcoin ETF hút vốn.",
                source_name="Src",
                published_date=MagicMock(
                    strftime=MagicMock(return_value="01/01/2026 10:00")
                ),
                url="https://example.com",
            )
        ]
        mock_llm = MagicMock()
        mock_llm.analyze.return_value = "# Báo cáo"

        use_case = GenerateMacroReportUseCase(
            config_reader=mock_config,
            news_fetcher=mock_fetcher,
            llm_client=mock_llm,
            # memory_manager=None (default)
        )
        result = use_case.execute()
        assert result == "# Báo cáo"
