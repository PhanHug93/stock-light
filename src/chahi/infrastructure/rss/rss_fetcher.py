"""RSS News Fetcher — implementation of INewsFetcher.

Cào tin tức từ RSS feeds sử dụng ``requests`` + ``feedparser``.

Bảo mật & chịu tải:
    - Timeout 15s cho mỗi HTTP request.
    - Retry 2 lần với exponential backoff khi gặp 5xx.
    - User-Agent header để tránh bị RSS servers block.
    - HTML cleaning bằng BeautifulSoup4 (an toàn, xử lý nested/malformed HTML).
    - Smart truncation tại word boundary tránh LLM token overflow.
"""

from __future__ import annotations

import calendar
import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser  # type: ignore[import-untyped]
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from chahi.core.entities import Article
from chahi.core.interfaces import INewsFetcher
from chahi.infrastructure.rss.rss_cache import RSSCache

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")

_REQUEST_TIMEOUT: int = 15  # seconds mỗi request
_MAX_SUMMARY_CHARS: int = 5000  # Phase 7: không cần chắt bóp token
_USER_AGENT: str = "ChaHi/0.1.0 RSS Fetcher (+https://github.com/chahi)"
_MAX_RETRIES: int = 2  # retry cho 5xx errors


# ── Helper functions ────────────────────────────────────────


def _build_session() -> requests.Session:
    """Tạo requests.Session với retry strategy.

    Retry 2 lần cho status 502/503/504, backoff 0.5s giữa các lần.
    Không retry cho 4xx (lỗi client, retry không giải quyết).

    Returns:
        Session đã được cấu hình retry.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=_MAX_RETRIES,
        backoff_factor=0.5,
        status_forcelist=[502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers["User-Agent"] = _USER_AGENT
    return session


def _smart_truncate(text: str, max_chars: int = _MAX_SUMMARY_CHARS) -> str:
    """Cắt chuỗi thông minh tại word boundary gần nhất.

    Thay vì cắt ngang từ (vd: "lạm ph..."), hàm này tìm khoảng trắng
    gần nhất trước ``max_chars`` để cắt toàn vẹn từ cuối cùng.

    Args:
        text: Chuỗi cần cắt.
        max_chars: Số ký tự tối đa cho phép.

    Returns:
        Chuỗi đã cắt + "…" nếu vượt max_chars, giữ nguyên nếu không.
    """
    if len(text) <= max_chars:
        return text

    # Tìm khoảng trắng gần nhất trước max_chars
    cut_point = text.rfind(" ", 0, max_chars)

    # Nếu không tìm thấy space (1 từ cực dài), cắt cứng
    if cut_point <= 0:
        cut_point = max_chars

    return text[:cut_point].rstrip() + "…"


class RSSNewsFetcher(INewsFetcher):
    """Cào tin tức từ RSS feeds qua requests + feedparser.

    Implement INewsFetcher — chịu trách nhiệm:
    1. Fetch RSS feed qua HTTP với timeout + retry.
    2. Parse response bằng feedparser.
    3. Map từng entry sang domain entity ``Article``.
    4. Clean HTML bằng BeautifulSoup4 và smart truncate summary.

    Args:
        source_name: Tên nguồn tin mặc định.
        timeout: Timeout giây cho mỗi HTTP request (default 15s).
    """

    def __init__(
        self,
        source_name: str = "Unknown",
        timeout: int = _REQUEST_TIMEOUT,
        cache: RSSCache | None = None,
    ) -> None:
        self._source_name = source_name
        self._timeout = timeout
        self._session = _build_session()
        self._cache = cache or RSSCache()

    def fetch_news(self, url: str, limit: int = 10) -> list[Article]:
        """Cào tin tức từ một RSS feed URL.

        Args:
            url: URL của RSS feed (chỉ http/https).
            limit: Số lượng bài viết tối đa cần lấy.

        Returns:
            Danh sách Article đã parse, tối đa ``limit`` bài.

        Raises:
            ConnectionError: Khi không thể kết nối hoặc HTTP error.
        """
        logger.info("Fetching RSS: %s (limit=%d, timeout=%ds)", url, limit, self._timeout)

        # ── Bước 1: HTTP request với timeout + retry ──
        raw_content = self._http_get(url)

        # ── Bước 2: Parse RSS ──
        feed = feedparser.parse(raw_content)

        if feed.bozo and not feed.entries:
            exception_msg = str(getattr(feed, "bozo_exception", "Unknown error"))
            raise ConnectionError(
                f"Không thể parse RSS feed: {url}. Lỗi: {exception_msg}"
            )

        # ── Bước 3: Map entries -> Articles ──
        articles: list[Article] = []
        entries = feed.entries[:limit]

        for entry in entries:
            article = self._parse_entry(entry)
            if article is not None:
                articles.append(article)

        logger.info(
            "Parsed %d/%d articles từ %s",
            len(articles),
            len(entries),
            url,
        )

        return articles

    # ── Private: HTTP ───────────────────────────────────────

    def _http_get(self, url: str) -> bytes:
        """Fetch URL với Conditional GET (ETag/Last-Modified).

        Flow:
        1. Gửi request với ``If-None-Match`` / ``If-Modified-Since``.
        2. Nếu server trả ``304 Not Modified`` → dùng cached content.
        3. Nếu ``200 OK`` → cập nhật cache với ETag/Last-Modified mới.

        Args:
            url: URL cần fetch.

        Returns:
            Raw response content (bytes).

        Raises:
            ConnectionError: Khi request thất bại.
        """
        # Lấy conditional headers từ cache
        extra_headers = self._cache.get_headers(url)

        try:
            response = self._session.get(
                url,
                timeout=self._timeout,
                headers=extra_headers,
            )

            # 304 Not Modified → dùng cache
            if response.status_code == 304:
                cached = self._cache.get_cached_content(url)
                if cached is not None:
                    logger.info("  304 Not Modified, dùng cache: %s", url)
                    return cached
                # Cache bị mất? Fetch lại bình thường
                logger.warning("  304 nhưng không có cache, fetch lại: %s", url)
                response = self._session.get(url, timeout=self._timeout)

            response.raise_for_status()

        except requests.Timeout:
            raise ConnectionError(
                f"RSS feed timeout sau {self._timeout}s: {url}"
            ) from None
        except requests.ConnectionError as exc:
            raise ConnectionError(
                f"Không thể kết nối tới RSS feed: {url}. Lỗi: {exc}"
            ) from None
        except requests.HTTPError as exc:
            raise ConnectionError(
                f"RSS feed trả về HTTP error: {url}. Lỗi: {exc}"
            ) from None
        except requests.RequestException as exc:
            raise ConnectionError(
                f"Lỗi không xác định khi fetch RSS: {url}. Lỗi: {exc}"
            ) from None

        # Cập nhật cache với ETag/Last-Modified mới
        self._cache.update(
            url=url,
            content=response.content,
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
        )

        return response.content

    # ── Private: Entry Parsing ──────────────────────────────

    def _parse_entry(self, entry: Any) -> Article | None:
        """Parse một RSS entry thành Article.

        Trả về None nếu entry thiếu title hoặc link.

        Args:
            entry: feedparser entry object.

        Returns:
            Article nếu parse thành công, None nếu entry không hợp lệ.
        """
        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()

        if not title or not link:
            logger.warning(
                "Bỏ qua entry thiếu title/link: title=%r, link=%r",
                title,
                link,
            )
            return None

        summary_raw = entry.get("summary", "") or entry.get("description", "") or ""
        summary = self._clean_html(summary_raw)

        # Smart truncate: cắt tại word boundary, không cắt ngang từ
        summary = _smart_truncate(summary, _MAX_SUMMARY_CHARS)

        published_date = self._parse_date(entry)

        return Article(
            title=title,
            summary=summary,
            source_name=self._source_name,
            published_date=published_date,
            url=link,
        )

    # ── Private: HTML Cleaning ──────────────────────────────

    @staticmethod
    def _clean_html(raw: str) -> str:
        """Làm sạch summary/description từ RSS entry bằng BeautifulSoup4.

        An toàn hơn regex — xử lý chính xác:
        - HTML lồng nhau phức tạp
        - Script/style tags (bị loại bỏ hoàn toàn)
        - HTML entities (tự động unescape)
        - Malformed HTML (BeautifulSoup tự sửa)

        Args:
            raw: Nội dung HTML thô từ RSS feed.

        Returns:
            Text đã được làm sạch, không chứa HTML.
        """
        if not raw:
            return ""

        # BeautifulSoup parse + extract text (tự unescape entities)
        soup = BeautifulSoup(raw, "html.parser")

        # Xóa script và style tags trước khi get_text
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ")

        # Normalize whitespace
        text = _WHITESPACE_RE.sub(" ", text)

        return text.strip()

    # ── Private: Date Parsing ───────────────────────────────

    @staticmethod
    def _parse_date(entry: Any) -> datetime:
        """Parse published date từ RSS entry.

        Thứ tự fallback:
        1. published_parsed (feedparser struct_time).
        2. published (raw RFC 2822 string).
        3. updated_parsed (Atom feeds).
        4. datetime.now(UTC).

        Args:
            entry: feedparser entry object.

        Returns:
            datetime với timezone (UTC).
        """
        parsed = entry.get("published_parsed")
        if parsed is not None:
            try:
                timestamp = calendar.timegm(parsed)
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            except (TypeError, ValueError, OverflowError):
                pass

        raw_date: str = entry.get("published", "")
        if raw_date:
            try:
                dt = parsedate_to_datetime(raw_date)
                return dt
            except (TypeError, ValueError):
                pass

        updated = entry.get("updated_parsed")
        if updated is not None:
            try:
                timestamp = calendar.timegm(updated)
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            except (TypeError, ValueError, OverflowError):
                pass

        logger.debug("Không parse được date, dùng UTC now.")
        return datetime.now(tz=timezone.utc)
