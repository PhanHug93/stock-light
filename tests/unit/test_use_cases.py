"""Unit tests cho GenerateMacroReportUseCase (Phase 7 — Map-Reduce).

Mock tất cả dependencies (IConfigReader, INewsFetcher, ILLMClient)
để test orchestration logic thuần túy.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from chahi.core.entities import (
    Article,
    SourceCategory,
    SourceConfig,
)
from chahi.core.use_cases import (
    MAP_PROMPT,
    REDUCE_PROMPT,
    GenerateMacroReportUseCase,
    _extract_summary,
)

# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────


def _make_article(
    title: str = "Test Article",
    source: str = "TestSource",
) -> Article:
    """Tạo Article mẫu."""
    return Article(
        title=title,
        summary=(
            "Fed giữ nguyên lãi suất, giá vàng ổn định và Bitcoin ETF duy trì dòng vốn."
        ),
        source_name=source,
        published_date=datetime(2026, 3, 18, 10, 0, tzinfo=UTC),
        url=f"https://example.com/{title.lower().replace(' ', '-')}",
    )


def _make_source(
    name: str = "Test Feed",
    url: str = "https://example.com/rss",
) -> SourceConfig:
    """Tạo SourceConfig mẫu."""
    return SourceConfig(name=name, url=url, type="rss")


@pytest.fixture()
def mock_config_reader() -> MagicMock:
    """Mock IConfigReader trả về 3 categories."""
    reader = MagicMock()
    reader.get_sources.return_value = {
        SourceCategory.OIL_MACRO: [_make_source("Reuters", "https://reuters.com/rss")],
        SourceCategory.GOLD: [_make_source("Kitco", "https://kitco.com/rss")],
        SourceCategory.CRYPTO: [_make_source("CoinDesk", "https://coindesk.com/rss")],
    }
    return reader


@pytest.fixture()
def mock_fetcher() -> MagicMock:
    """Mock INewsFetcher trả về 1 article cho mỗi URL."""
    fetcher = MagicMock()
    fetcher.fetch_news.return_value = [_make_article(title="Article 1")]
    return fetcher


@pytest.fixture()
def mock_llm() -> MagicMock:
    """Mock ILLMClient trả về báo cáo."""
    llm = MagicMock()
    llm.analyze.return_value = "# Báo cáo\nNội dung phân tích..."
    return llm


# ─────────────────────────────────────────────────────────────
# execute() — Map-Reduce Pipeline
# ─────────────────────────────────────────────────────────────


class TestExecuteMapReduce:
    """Tests cho Map-Reduce pipeline trong execute()."""

    def test_returns_llm_response(
        self,
        mock_config_reader: MagicMock,
        mock_fetcher: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        """execute() phải trả về response từ LLM."""
        use_case = GenerateMacroReportUseCase(
            config_reader=mock_config_reader,
            news_fetcher=mock_fetcher,
            llm_client=mock_llm,
        )
        result = use_case.execute()

        assert result == "# Báo cáo\nNội dung phân tích..."

    def test_calls_config_reader(
        self,
        mock_config_reader: MagicMock,
        mock_fetcher: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        """execute() phải gọi get_sources() từ config reader."""
        use_case = GenerateMacroReportUseCase(
            config_reader=mock_config_reader,
            news_fetcher=mock_fetcher,
            llm_client=mock_llm,
        )
        use_case.execute()

        mock_config_reader.get_sources.assert_called_once()

    def test_fetches_from_all_sources(
        self,
        mock_config_reader: MagicMock,
        mock_fetcher: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        """execute() phải gọi fetch_news() cho mỗi source URL."""
        use_case = GenerateMacroReportUseCase(
            config_reader=mock_config_reader,
            news_fetcher=mock_fetcher,
            llm_client=mock_llm,
        )
        use_case.execute()

        # 3 categories × 1 source mỗi category = 3 fetch calls
        assert mock_fetcher.fetch_news.call_count == 3

    def test_map_calls_llm_per_category(
        self,
        mock_config_reader: MagicMock,
        mock_fetcher: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        """MAP phải gọi LLM 3 lần (mỗi category) + 1 lần REDUCE = 4 calls."""
        use_case = GenerateMacroReportUseCase(
            config_reader=mock_config_reader,
            news_fetcher=mock_fetcher,
            llm_client=mock_llm,
        )
        use_case.execute()

        # 3 MAP calls + 1 REDUCE call = 4
        assert mock_llm.analyze.call_count == 4

    def test_reduce_uses_reduce_prompt(
        self,
        mock_config_reader: MagicMock,
        mock_fetcher: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        """REDUCE step phải dùng REDUCE_PROMPT."""
        use_case = GenerateMacroReportUseCase(
            config_reader=mock_config_reader,
            news_fetcher=mock_fetcher,
            llm_client=mock_llm,
        )
        use_case.execute()

        # Lần gọi cuối cùng là REDUCE
        last_call = mock_llm.analyze.call_args_list[-1]
        assert last_call[1]["system_prompt"] == REDUCE_PROMPT

    def test_map_input_uses_semantic_article_tags(
        self,
        mock_config_reader: MagicMock,
        mock_fetcher: MagicMock,
    ) -> None:
        """MAP input phải dùng thẻ <article> để giảm nhiễu parser."""
        mock_llm = MagicMock()
        mock_llm.supports_concurrency = False
        mock_llm.analyze.side_effect = [
            "- MAP Oil",
            "- MAP Gold",
            "- MAP Crypto",
            "## 6. 📋 Tổng kết\n- Kết luận",
        ]

        use_case = GenerateMacroReportUseCase(
            config_reader=mock_config_reader,
            news_fetcher=mock_fetcher,
            llm_client=mock_llm,
        )
        use_case.execute()

        first_map_call = mock_llm.analyze.call_args_list[0]
        map_user_content = first_map_call[1]["user_content"]
        assert "<article id=" in map_user_content
        assert "<summary>" in map_user_content
        assert "</category>" in map_user_content

    def test_reduce_input_contains_previous_lessons_tag(self) -> None:
        """REDUCE input phải chứa thẻ previous_lessons khi memory trả dữ liệu."""
        mock_config = MagicMock()
        mock_config.get_sources.return_value = {
            SourceCategory.OIL_MACRO: [
                _make_source("Reuters", "https://reuters.com/rss")
            ],
        }

        mock_fetcher = MagicMock()
        mock_fetcher.fetch_news.return_value = [_make_article(title="Fed Policy")]

        mock_llm = MagicMock()
        mock_llm.supports_concurrency = False
        mock_llm.analyze.side_effect = [
            "- MAP Oil",
            "## 6. 📋 Tổng kết\n- Kết luận",
        ]

        mock_memory = MagicMock()
        mock_memory.retrieve_last_context.return_value = "Bài học hôm trước"
        mock_memory.retrieve_related_context.return_value = "Bài học cùng mẫu hình Fed"

        use_case = GenerateMacroReportUseCase(
            config_reader=mock_config,
            news_fetcher=mock_fetcher,
            llm_client=mock_llm,
            memory_manager=mock_memory,
        )
        use_case.execute()

        mock_memory.retrieve_related_context.assert_called_once()
        reduce_call = mock_llm.analyze.call_args_list[-1]
        reduce_user_content = reduce_call[1]["user_content"]
        assert "<previous_lessons>" in reduce_user_content
        assert "[BÀI HỌC LIÊN QUAN]" in reduce_user_content


# ─────────────────────────────────────────────────────────────
# execute() — Edge Cases
# ─────────────────────────────────────────────────────────────


class TestExecuteEdgeCases:
    """Tests cho execute() — edge cases."""

    def test_empty_news_returns_warning(
        self,
        mock_config_reader: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        """Khi không fetch được tin nào, trả về warning message."""
        empty_fetcher = MagicMock()
        empty_fetcher.fetch_news.return_value = []

        use_case = GenerateMacroReportUseCase(
            config_reader=mock_config_reader,
            news_fetcher=empty_fetcher,
            llm_client=mock_llm,
        )
        result = use_case.execute()

        assert "Không có tin tức" in result
        mock_llm.analyze.assert_not_called()

    def test_partial_fetch_failure_continues(
        self,
        mock_config_reader: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        """Khi một source lỗi, vẫn tiếp tục fetch các source khác."""
        partial_fetcher = MagicMock()

        def side_effect(url: str, limit: int = 10) -> list[Article]:
            if "reuters" in url:
                raise ConnectionError("Timeout")
            return [_make_article()]

        partial_fetcher.fetch_news.side_effect = side_effect

        use_case = GenerateMacroReportUseCase(
            config_reader=mock_config_reader,
            news_fetcher=partial_fetcher,
            llm_client=mock_llm,
        )
        use_case.execute()

        # LLM vẫn được gọi vì có tin từ gold + crypto
        assert mock_llm.analyze.call_count >= 1


# ─────────────────────────────────────────────────────────────
# MAP_PROMPT & REDUCE_PROMPT
# ─────────────────────────────────────────────────────────────


class TestPrompts:
    """Tests cho MAP_PROMPT và REDUCE_PROMPT."""

    def test_map_prompt_non_empty(self) -> None:
        """MAP_PROMPT phải có nội dung."""
        assert len(MAP_PROMPT) > 50

    def test_map_prompt_has_placeholder(self) -> None:
        """MAP_PROMPT phải có placeholder cho category_name."""
        assert "{category_name}" in MAP_PROMPT

    def test_reduce_prompt_non_empty(self) -> None:
        """REDUCE_PROMPT phải có nội dung."""
        assert len(REDUCE_PROMPT) > 100

    def test_reduce_prompt_contains_key_sections(self) -> None:
        """REDUCE_PROMPT phải chứa các phần quan trọng."""
        assert "Dầu" in REDUCE_PROMPT or "dầu" in REDUCE_PROMPT
        assert "Vàng" in REDUCE_PROMPT or "vàng" in REDUCE_PROMPT
        assert "Crypto" in REDUCE_PROMPT or "crypto" in REDUCE_PROMPT
        assert "Ngân hàng" in REDUCE_PROMPT or "ngân hàng" in REDUCE_PROMPT

    def test_reduce_prompt_vietnam_focus(self) -> None:
        """REDUCE_PROMPT phải có trọng tâm Việt Nam."""
        assert "VN-Index" in REDUCE_PROMPT or "Việt Nam" in REDUCE_PROMPT

    def test_reduce_prompt_has_summary_marker(self) -> None:
        """REDUCE_PROMPT phải yêu cầu format Tổng kết cho Memory."""
        assert "Tổng kết" in REDUCE_PROMPT


# ─────────────────────────────────────────────────────────────
# _extract_summary
# ─────────────────────────────────────────────────────────────


class TestExtractSummary:
    """Tests cho _extract_summary()."""

    def test_extracts_summary_section(self) -> None:
        """Phải trích xuất phần sau '## 6. 📋 Tổng kết'."""
        report = "Some content\n## 6. 📋 Tổng kết\n- Ý 1\n- Ý 2"
        result = _extract_summary(report)
        assert "Ý 1" in result
        assert "Ý 2" in result

    def test_fallback_when_no_summary(self) -> None:
        """Khi không có section Tổng kết, lấy 500 chars cuối."""
        report = "A very long report " * 50
        result = _extract_summary(report)
        assert len(result) <= 500
