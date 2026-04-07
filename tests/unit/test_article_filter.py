"""Unit tests cho ArticleFilterService."""

from __future__ import annotations

from datetime import UTC, datetime

from chahi.core.entities import AnalysisContext, Article, SourceCategory
from chahi.core.services.article_filter import ArticleFilterService


def _article(title: str, summary: str, url: str) -> Article:
    return Article(
        title=title,
        summary=summary,
        source_name="TestSource",
        published_date=datetime(2026, 4, 6, 10, 0, tzinfo=UTC),
        url=url,
    )


class TestArticleFilterService:
    """Tests cho keyword filtering + Jaccard dedup."""

    def test_keyword_filter_removes_irrelevant_article(self) -> None:
        service = ArticleFilterService(similarity_threshold=0.70)
        relevant = _article(
            title="Fed giữ nguyên lãi suất tháng 4",
            summary="Thị trường kỳ vọng chính sách tiền tệ ổn định.",
            url="https://example.com/fed",
        )
        irrelevant = _article(
            title="Top 10 bộ phim hay cuối tuần",
            summary="Danh sách phim giải trí mới cập nhật.",
            url="https://example.com/movie",
        )

        filtered, stats = service.filter_articles(
            SourceCategory.OIL_MACRO,
            [relevant, irrelevant],
        )

        assert len(filtered) == 1
        assert filtered[0].url == "https://example.com/fed"
        assert stats.input_count == 2
        assert stats.keyword_kept == 1

    def test_jaccard_dedup_keeps_longer_article(self) -> None:
        service = ArticleFilterService(similarity_threshold=0.70)
        short_ver = _article(
            title="Fed giữ nguyên lãi suất, DXY đi ngang",
            summary="Fed giữ nguyên lãi suất trong cuộc họp mới nhất.",
            url="https://example.com/short",
        )
        long_ver = _article(
            title="Fed giữ nguyên lãi suất, DXY đi ngang",
            summary=(
                "Fed giữ nguyên lãi suất trong cuộc họp mới nhất. "
                "Fed giữ nguyên lãi suất trong cuộc họp mới nhất "
                "với giọng điệu thận trọng."
            ),
            url="https://example.com/long",
        )

        filtered, stats = service.filter_articles(
            SourceCategory.OIL_MACRO,
            [short_ver, long_ver],
        )

        assert len(filtered) == 1
        assert filtered[0].url == "https://example.com/long"
        assert stats.dedup_removed == 1

    def test_extract_hot_keywords(self) -> None:
        service = ArticleFilterService()
        context = AnalysisContext(
            date=datetime(2026, 4, 6, tzinfo=UTC).date(),
            news_by_category={
                SourceCategory.OIL_MACRO: [
                    _article(
                        title="Fed phát tín hiệu giữ lãi suất",
                        summary="Fed theo dõi CPI và DXY chặt chẽ.",
                        url="https://example.com/oil",
                    )
                ],
                SourceCategory.GOLD: [
                    _article(
                        title="Giá vàng tăng khi DXY suy yếu",
                        summary="Gold ETF hút vốn trở lại.",
                        url="https://example.com/gold",
                    )
                ],
                SourceCategory.CRYPTO: [
                    _article(
                        title="Bitcoin ETF duy trì dòng vốn vào",
                        summary="BTC và ETH đồng thuận tăng.",
                        url="https://example.com/crypto",
                    )
                ],
            },
        )

        hot_keywords = service.extract_hot_keywords(context, max_keywords=5)

        assert hot_keywords
        assert "fed" in hot_keywords or "bitcoin" in hot_keywords
