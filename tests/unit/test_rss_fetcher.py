"""Unit tests cho RSSNewsFetcher.

Sử dụng unittest.mock để mock requests.Session và feedparser,
không cần kết nối internet khi test.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from chahi.core.entities import Article
from chahi.infrastructure.rss.rss_fetcher import RSSNewsFetcher

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _make_entry(
    title: str = "Test Article",
    link: str = "https://example.com/article",
    summary: str = "Test summary",
    published: str = "Wed, 18 Mar 2026 10:00:00 GMT",
    published_parsed: tuple[int, ...] | None = (2026, 3, 18, 10, 0, 0, 2, 77, 0),
) -> dict[str, Any]:
    """Tạo mock RSS entry giống feedparser output."""
    entry: dict[str, Any] = {
        "title": title,
        "link": link,
        "summary": summary,
        "published": published,
    }
    if published_parsed is not None:
        entry["published_parsed"] = time.struct_time(published_parsed)
    return entry


def _make_feed(
    entries: list[dict[str, Any]] | None = None,
    bozo: bool = False,
    bozo_exception: str | None = None,
) -> SimpleNamespace:
    """Tạo mock feedparser result."""
    feed = SimpleNamespace(
        entries=entries or [],
        bozo=bozo,
    )
    if bozo_exception is not None:
        feed.bozo_exception = bozo_exception  # type: ignore[attr-defined]
    return feed


def _make_http_response(
    content: bytes = b"<rss>ok</rss>", status: int = 200
) -> MagicMock:
    """Tạo mock requests.Response."""
    resp = MagicMock()
    resp.content = content
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}")
    return resp


# ─────────────────────────────────────────────────────────────
# HTML Cleaning
# ─────────────────────────────────────────────────────────────


class TestCleanHTML:
    """Tests cho _clean_html() helper."""

    def test_strips_html_tags(self) -> None:
        fetcher = RSSNewsFetcher(source_name="Test")
        result = fetcher._clean_html("<p>Hello <b>world</b></p>")
        assert result == "Hello world"

    def test_strips_links(self) -> None:
        fetcher = RSSNewsFetcher(source_name="Test")
        result = fetcher._clean_html('<a href="https://example.com">Click here</a>')
        assert result == "Click here"

    def test_unescapes_entities(self) -> None:
        fetcher = RSSNewsFetcher(source_name="Test")
        result = fetcher._clean_html("Oil &amp; Gas &#8212; $100")
        assert result == "Oil & Gas — $100"

    def test_normalizes_whitespace(self) -> None:
        fetcher = RSSNewsFetcher(source_name="Test")
        result = fetcher._clean_html("  Hello   \n\n  world  ")
        assert result == "Hello world"

    def test_empty_string(self) -> None:
        fetcher = RSSNewsFetcher(source_name="Test")
        assert fetcher._clean_html("") == ""

    def test_complex_rss_summary(self) -> None:
        fetcher = RSSNewsFetcher(source_name="Test")
        raw = (
            '<div class="feed-description">'
            "<p>Oil prices fell <b>2%</b> on Monday.</p>"
            '<img src="foo.jpg" />'
            "<br/>&amp;more details"
            "</div>"
        )
        result = fetcher._clean_html(raw)
        assert "Oil prices fell" in result
        assert "2%" in result
        assert "&more details" in result
        assert "<" not in result


# ─────────────────────────────────────────────────────────────
# fetch_news() — Happy Path
# ─────────────────────────────────────────────────────────────


class TestFetchNewsValid:
    """Tests cho fetch_news() với feed hợp lệ."""

    @patch("chahi.infrastructure.rss.rss_fetcher.RSSCache")
    @patch("chahi.infrastructure.rss.rss_fetcher.feedparser.parse")
    @patch("chahi.infrastructure.rss.rss_fetcher._build_session")
    def test_returns_articles(
        self, mock_session_fn: Any, mock_parse: Any, mock_cache_cls: Any
    ) -> None:
        mock_session = MagicMock()
        mock_session.get.return_value = _make_http_response()
        mock_session_fn.return_value = mock_session
        mock_parse.return_value = _make_feed(
            entries=[_make_entry(), _make_entry(title="Article 2")]
        )

        fetcher = RSSNewsFetcher(source_name="Reuters")
        articles = fetcher.fetch_news("https://example.com/rss")

        assert len(articles) == 2
        assert all(isinstance(a, Article) for a in articles)

    @patch("chahi.infrastructure.rss.rss_fetcher.RSSCache")
    @patch("chahi.infrastructure.rss.rss_fetcher.feedparser.parse")
    @patch("chahi.infrastructure.rss.rss_fetcher._build_session")
    def test_maps_fields_correctly(
        self, mock_session_fn: Any, mock_parse: Any, mock_cache_cls: Any
    ) -> None:
        mock_session = MagicMock()
        mock_session.get.return_value = _make_http_response()
        mock_session_fn.return_value = mock_session
        mock_parse.return_value = _make_feed(
            entries=[
                _make_entry(
                    title="Fed Rate Decision",
                    link="https://reuters.com/fed",
                    summary="<p>The Fed held rates steady.</p>",
                )
            ]
        )

        fetcher = RSSNewsFetcher(source_name="Reuters")
        articles = fetcher.fetch_news("https://example.com/rss")

        article = articles[0]
        assert article.title == "Fed Rate Decision"
        assert article.url == "https://reuters.com/fed"
        assert article.summary == "The Fed held rates steady."
        assert article.source_name == "Reuters"

    @patch("chahi.infrastructure.rss.rss_fetcher.RSSCache")
    @patch("chahi.infrastructure.rss.rss_fetcher.feedparser.parse")
    @patch("chahi.infrastructure.rss.rss_fetcher._build_session")
    def test_respects_limit(
        self, mock_session_fn: Any, mock_parse: Any, mock_cache_cls: Any
    ) -> None:
        mock_session = MagicMock()
        mock_session.get.return_value = _make_http_response()
        mock_session_fn.return_value = mock_session
        mock_parse.return_value = _make_feed(
            entries=[_make_entry(title=f"Article {i}") for i in range(20)]
        )

        fetcher = RSSNewsFetcher(source_name="Test")
        articles = fetcher.fetch_news("https://example.com/rss", limit=5)

        assert len(articles) == 5

    @patch("chahi.infrastructure.rss.rss_fetcher.RSSCache")
    @patch("chahi.infrastructure.rss.rss_fetcher.feedparser.parse")
    @patch("chahi.infrastructure.rss.rss_fetcher._build_session")
    def test_skips_entry_without_title(
        self, mock_session_fn: Any, mock_parse: Any, mock_cache_cls: Any
    ) -> None:
        mock_session = MagicMock()
        mock_session.get.return_value = _make_http_response()
        mock_session_fn.return_value = mock_session
        mock_parse.return_value = _make_feed(
            entries=[
                _make_entry(title="", link="https://example.com/1"),
                _make_entry(title="Valid Article"),
            ]
        )

        fetcher = RSSNewsFetcher(source_name="Test")
        articles = fetcher.fetch_news("https://example.com/rss")

        assert len(articles) == 1
        assert articles[0].title == "Valid Article"

    @patch("chahi.infrastructure.rss.rss_fetcher.RSSCache")
    @patch("chahi.infrastructure.rss.rss_fetcher.feedparser.parse")
    @patch("chahi.infrastructure.rss.rss_fetcher._build_session")
    def test_truncates_long_summary(
        self, mock_session_fn: Any, mock_parse: Any, mock_cache_cls: Any
    ) -> None:
        """Summary vượt token budget phải bị truncate."""
        mock_session = MagicMock()
        mock_session.get.return_value = _make_http_response()
        mock_session_fn.return_value = mock_session
        # Realistic text: each word = ~1 token, need > 1500 tokens
        long_summary = "Giá dầu tăng mạnh hôm nay " * 500  # ~3500 tokens
        mock_parse.return_value = _make_feed(
            entries=[_make_entry(summary=long_summary)]
        )

        fetcher = RSSNewsFetcher(source_name="Test")
        articles = fetcher.fetch_news("https://example.com/rss")

        assert len(articles[0].summary) < len(long_summary)
        assert articles[0].summary.endswith("…")


# ─────────────────────────────────────────────────────────────
# fetch_news() — Error Cases
# ─────────────────────────────────────────────────────────────


class TestFetchNewsErrors:
    """Tests cho fetch_news() — error handling."""

    @patch("chahi.infrastructure.rss.rss_fetcher._build_session")
    def test_http_timeout(self, mock_session_fn: Any) -> None:
        """HTTP timeout phải raise ConnectionError."""
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.Timeout("timed out")
        mock_session_fn.return_value = mock_session

        fetcher = RSSNewsFetcher(source_name="Test")
        with pytest.raises(ConnectionError, match="timeout"):
            fetcher.fetch_news("https://example.com/rss")

    @patch("chahi.infrastructure.rss.rss_fetcher._build_session")
    def test_connection_error(self, mock_session_fn: Any) -> None:
        """Connection error phải raise ConnectionError."""
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.ConnectionError("refused")
        mock_session_fn.return_value = mock_session

        fetcher = RSSNewsFetcher(source_name="Test")
        with pytest.raises(ConnectionError, match="kết nối"):
            fetcher.fetch_news("https://example.com/rss")

    @patch("chahi.infrastructure.rss.rss_fetcher._build_session")
    def test_http_error(self, mock_session_fn: Any) -> None:
        """HTTP 404/500 phải raise ConnectionError."""
        mock_session = MagicMock()
        mock_session.get.return_value = _make_http_response(status=404)
        mock_session_fn.return_value = mock_session

        fetcher = RSSNewsFetcher(source_name="Test")
        with pytest.raises(ConnectionError, match="HTTP"):
            fetcher.fetch_news("https://example.com/rss")

    @patch("chahi.infrastructure.rss.rss_fetcher.RSSCache")
    @patch("chahi.infrastructure.rss.rss_fetcher.feedparser.parse")
    @patch("chahi.infrastructure.rss.rss_fetcher._build_session")
    def test_bozo_with_no_entries(
        self, mock_session_fn: Any, mock_parse: Any, mock_cache_cls: Any
    ) -> None:
        """Feed malformed + no entries phải raise ConnectionError."""
        mock_session = MagicMock()
        mock_session.get.return_value = _make_http_response()
        mock_session_fn.return_value = mock_session
        mock_parse.return_value = _make_feed(
            bozo=True, bozo_exception="SAXParseException"
        )

        fetcher = RSSNewsFetcher(source_name="Test")
        with pytest.raises(ConnectionError, match="SAXParseException"):
            fetcher.fetch_news("https://example.com/rss")

    @patch("chahi.infrastructure.rss.rss_fetcher.RSSCache")
    @patch("chahi.infrastructure.rss.rss_fetcher.feedparser.parse")
    @patch("chahi.infrastructure.rss.rss_fetcher._build_session")
    def test_bozo_with_entries_still_works(
        self, mock_session_fn: Any, mock_parse: Any, mock_cache_cls: Any
    ) -> None:
        """Feed bozo nhưng có entries → vẫn parse bình thường."""
        mock_session = MagicMock()
        mock_session.get.return_value = _make_http_response()
        mock_session_fn.return_value = mock_session
        mock_parse.return_value = _make_feed(entries=[_make_entry()], bozo=True)

        fetcher = RSSNewsFetcher(source_name="Test")
        articles = fetcher.fetch_news("https://example.com/rss")

        assert len(articles) == 1


# ─────────────────────────────────────────────────────────────
# URL Scheme Validation (anti-SSRF)
# ─────────────────────────────────────────────────────────────


class TestURLSchemeValidation:
    """Tests cho SourceConfig URL scheme validation."""

    def test_rejects_file_scheme(self) -> None:
        """file:// URL phải bị reject."""
        from chahi.core.entities import SourceConfig

        with pytest.raises(ValueError, match="http hoặc https"):
            SourceConfig(name="Evil", url="file:///etc/passwd")

    def test_rejects_ftp_scheme(self) -> None:
        """ftp:// URL phải bị reject."""
        from chahi.core.entities import SourceConfig

        with pytest.raises(ValueError, match="http hoặc https"):
            SourceConfig(name="Evil", url="ftp://example.com/file")

    def test_allows_https(self) -> None:
        """https:// URL phải pass."""
        from chahi.core.entities import SourceConfig

        src = SourceConfig(name="Good", url="https://reuters.com/rss")
        assert src.url == "https://reuters.com/rss"

    def test_allows_http(self) -> None:
        """http:// URL phải pass (cần cho local testing)."""
        from chahi.core.entities import SourceConfig

        src = SourceConfig(name="Local", url="http://localhost:3000/rss")
        assert src.url == "http://localhost:3000/rss"


# ─────────────────────────────────────────────────────────────
# Deep Scraper — _fetch_full_text() & Teaser Detection
# ─────────────────────────────────────────────────────────────


class TestDeepScraper:
    """Tests cho Phase 10: Deep Web Scraper & Full-text Extraction."""

    @patch("chahi.infrastructure.rss.rss_fetcher.RSSCache")
    @patch("chahi.infrastructure.rss.rss_fetcher._trafilatura_extract")
    @patch("chahi.infrastructure.rss.rss_fetcher._build_session")
    def test_fetch_full_text_success(
        self,
        mock_session_fn: Any,
        mock_extract: Any,
        mock_cache_cls: Any,
    ) -> None:
        """_fetch_full_text trả về text khi extract thành công."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "<html><body>Full article</body></html>"
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session_fn.return_value = mock_session

        mock_extract.return_value = "Full article content here."

        fetcher = RSSNewsFetcher(source_name="Test")
        # Mock the CPU pool to call function directly (avoid pickle issues)
        fetcher._cpu_pool = MagicMock()
        future_mock = MagicMock()
        future_mock.result.return_value = "Full article content here."
        fetcher._cpu_pool.schedule.return_value = future_mock

        result = fetcher._fetch_full_text("https://example.com/article")

        assert result == "Full article content here."

    @patch("chahi.infrastructure.rss.rss_fetcher.RSSCache")
    @patch("chahi.infrastructure.rss.rss_fetcher._build_session")
    def test_fetch_full_text_timeout(
        self,
        mock_session_fn: Any,
        mock_cache_cls: Any,
    ) -> None:
        """_fetch_full_text trả về None khi timeout."""
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.Timeout("timed out")
        mock_session_fn.return_value = mock_session

        fetcher = RSSNewsFetcher(source_name="Test")
        result = fetcher._fetch_full_text("https://example.com/article")

        assert result is None

    @patch("chahi.infrastructure.rss.rss_fetcher.RSSCache")
    @patch("chahi.infrastructure.rss.rss_fetcher._build_session")
    def test_fetch_full_text_http_403(
        self,
        mock_session_fn: Any,
        mock_cache_cls: Any,
    ) -> None:
        """_fetch_full_text trả về None khi bị 403 Forbidden."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 403
        http_error = requests.HTTPError(response=mock_response)
        mock_response.raise_for_status.side_effect = http_error
        mock_session.get.return_value = mock_response
        mock_session_fn.return_value = mock_session

        fetcher = RSSNewsFetcher(source_name="Test")
        result = fetcher._fetch_full_text("https://example.com/article")

        assert result is None

    @patch("chahi.infrastructure.rss.rss_fetcher.RSSCache")
    @patch("chahi.infrastructure.rss.rss_fetcher.trafilatura.extract")
    @patch("chahi.infrastructure.rss.rss_fetcher._build_session")
    def test_fetch_full_text_extract_returns_none(
        self,
        mock_session_fn: Any,
        mock_extract: Any,
        mock_cache_cls: Any,
    ) -> None:
        """_fetch_full_text trả về None khi trafilatura extract trả None."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "<html></html>"
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session_fn.return_value = mock_session

        mock_extract.return_value = None

        fetcher = RSSNewsFetcher(source_name="Test")
        # Mock CPU pool
        fetcher._cpu_pool = MagicMock()
        future_mock = MagicMock()
        future_mock.result.return_value = None
        fetcher._cpu_pool.schedule.return_value = future_mock

        result = fetcher._fetch_full_text("https://example.com/article")

        assert result is None

    @patch("chahi.infrastructure.rss.rss_fetcher.RSSCache")
    @patch("chahi.infrastructure.rss.rss_fetcher.feedparser.parse")
    @patch("chahi.infrastructure.rss.rss_fetcher._build_session")
    def test_parse_entry_triggers_deep_scraper_on_short_summary(
        self,
        mock_session_fn: Any,
        mock_parse: Any,
        mock_cache_cls: Any,
    ) -> None:
        """Summary < 500 chars phải kích hoạt Deep Scraper."""
        mock_session = MagicMock()
        # First call = RSS fetch
        mock_session.get.return_value = _make_http_response()
        mock_session_fn.return_value = mock_session

        # RSS entry with short summary (teaser)
        short_summary = "Breaking news about oil prices."
        mock_parse.return_value = _make_feed(
            entries=[_make_entry(summary=short_summary)]
        )

        full_text = "Full article " * 100  # ~1300 chars

        fetcher = RSSNewsFetcher(source_name="Test")
        # Mock CPU pool to avoid pickle issues
        fetcher._cpu_pool = MagicMock()
        future_mock = MagicMock()
        future_mock.result.return_value = full_text.strip()
        fetcher._cpu_pool.schedule.return_value = future_mock
        # Also need to mock the second session.get for Deep Scraper
        deep_resp = MagicMock()
        deep_resp.text = "<html>article</html>"
        deep_resp.raise_for_status = MagicMock()
        mock_session.get.side_effect = [
            _make_http_response(),  # RSS fetch
            deep_resp,  # Deep scraper fetch
        ]

        articles = fetcher.fetch_news("https://example.com/rss")

        assert len(articles) == 1
        # Summary should be replaced with full text
        assert articles[0].summary != short_summary
        assert "Full article" in articles[0].summary

    @patch("chahi.infrastructure.rss.rss_fetcher.RSSCache")
    @patch("chahi.infrastructure.rss.rss_fetcher.feedparser.parse")
    @patch("chahi.infrastructure.rss.rss_fetcher._build_session")
    def test_parse_entry_skips_deep_scraper_on_long_summary(
        self,
        mock_session_fn: Any,
        mock_parse: Any,
        mock_cache_cls: Any,
    ) -> None:
        """Summary >= 500 chars KHÔNG kích hoạt Deep Scraper."""
        mock_session = MagicMock()
        mock_session.get.return_value = _make_http_response()
        mock_session_fn.return_value = mock_session

        # Long enough summary (no teaser)
        long_summary = "A" * 600
        mock_parse.return_value = _make_feed(
            entries=[_make_entry(summary=long_summary)]
        )

        fetcher = RSSNewsFetcher(source_name="Test")
        articles = fetcher.fetch_news("https://example.com/rss")

        assert len(articles) == 1
        assert articles[0].summary == long_summary

    @patch("chahi.infrastructure.rss.rss_fetcher.RSSCache")
    @patch("chahi.infrastructure.rss.rss_fetcher.feedparser.parse")
    @patch("chahi.infrastructure.rss.rss_fetcher._build_session")
    def test_parse_entry_fallback_on_deep_scraper_failure(
        self,
        mock_session_fn: Any,
        mock_parse: Any,
        mock_cache_cls: Any,
    ) -> None:
        """Deep Scraper thất bại → fallback dùng RSS summary cũ."""
        mock_session = MagicMock()
        mock_http_resp = _make_http_response()
        # First call = RSS fetch, second call = deep scraper (timeout)
        mock_session.get.side_effect = [
            mock_http_resp,
            requests.Timeout("timed out"),
        ]
        mock_session_fn.return_value = mock_session

        short_summary = "Short teaser."
        mock_parse.return_value = _make_feed(
            entries=[_make_entry(summary=short_summary)]
        )

        fetcher = RSSNewsFetcher(source_name="Test")
        articles = fetcher.fetch_news("https://example.com/rss")

        assert len(articles) == 1
        # Should fallback to original short summary
        assert articles[0].summary == short_summary
