"""RSS News Fetcher — implementation of INewsFetcher.

Hỗ trợ 2 chế độ:
1. ``fetch_news()`` sync cho từng feed đơn lẻ (giữ tương thích cũ).
2. ``fetch_many()`` async batch qua ``httpx.AsyncClient`` cho crawling diện rộng.

Bảo mật & chịu tải:
    - Timeout + retry có backoff cho lỗi mạng/5xx/429.
    - Manual redirect với kiểm tra SSRF target.
    - Semaphore + jitter delay để tránh burst gây rate-limit.
    - Deep Scraper khi summary RSS quá ngắn.
    - Fallback ``curl_cffi`` khi deep scraper gặp HTTP 403.
    - Full-text disk cache (diskcache) cho URL bài viết đã scrape.
"""

from __future__ import annotations

import asyncio
import calendar
import importlib
import ipaddress
import logging
import random
import re
import socket
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse

import feedparser  # type: ignore[import-untyped]
import httpx
import requests
import trafilatura
from bs4 import BeautifulSoup
from pebble import ProcessPool
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from chahi.core.entities import Article
from chahi.core.interfaces import INewsFetcher
from chahi.infrastructure.llm.token_counter import truncate_to_token_budget
from chahi.infrastructure.rss.circuit_breaker import BreakerState, DomainCircuitBreaker
from chahi.infrastructure.rss.rss_cache import RSSCache
from chahi.infrastructure.rss.scraper_policy import (
    canonicalize_url,
    is_allowed_by_policy,
    is_google_news_url,
    score_relevancy,
)

if TYPE_CHECKING:
    from chahi.core.entities import SourceCategory, SourceConfig

curl_cffi_requests: Any | None = None

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")

_REQUEST_TIMEOUT: int = 15  # seconds mỗi request
_MAX_SUMMARY_TOKENS: int = 1500  # Token budget per article summary
_MIN_SUMMARY_THRESHOLD: int = 500  # Phase 10: ngưỡng kích hoạt Deep Scraper
_USER_AGENT: str = "ChaHi/0.0.5 RSS Fetcher (+https://github.com/chahi)"
_MAX_RETRIES: int = 2  # retry cho 5xx errors
_CPU_WORKERS: int = 2  # ProcessPool workers cho trafilatura
_CIRCUIT_BREAKER_THRESHOLD: int = 2  # Sau N failures liên tiếp → ngắt domain
_CIRCUIT_BREAKER_OPEN_TTL_SECONDS: float = 900.0
_MAX_REDIRECTS: int = 5  # Tối đa redirect hops (chống SSRF)
_ASYNC_MAX_CONNECTIONS: int = 15  # Global semaphore
_ASYNC_JITTER_MIN: float = 0.5  # Per-domain delay min (seconds)
_ASYNC_JITTER_MAX: float = 2.5  # Per-domain delay max (seconds)
_ASYNC_CONNECT_TIMEOUT: float = 5.0
_ASYNC_READ_TIMEOUT: float = 15.0
_ASYNC_WRITE_TIMEOUT: float = 5.0
_ASYNC_POOL_TIMEOUT: float = 5.0
_ASYNC_MAX_RETRIES: int = 3
_FULL_TEXT_CACHE_DIR = Path(".cache/fulltext")
_ASYNC_RETRYABLE_STATUS_CODES: set[int] = {429, 502, 503}
_DEEP_SCRAPE_MAX_PER_FEED: int = 8
_GOOGLE_RESOLVE_TIMEOUT_SECONDS: int = 6
_NUMERIC_TOKEN_RE = re.compile(r"[+-]?\d+(?:[.,]\d+)?%?")


# ── Helper functions ────────────────────────────────────────


def _build_session() -> requests.Session:
    """Tạo requests.Session với retry strategy.

    Retry 2 lần cho status 429/502/503/504, backoff 0.5s giữa các lần.
    Tôn trọng header ``Retry-After`` từ server (429 rate limit).
    Không retry cho 4xx khác (lỗi client, retry không giải quyết).

    Returns:
        Session đã được cấu hình retry.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=_MAX_RETRIES,
        backoff_factor=0.5,
        status_forcelist=[429, 502, 503, 504],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers["User-Agent"] = _USER_AGENT
    return session


def _is_ssrf_target(hostname: str) -> bool:
    """Kiểm tra hostname có resolve tới IP private/loopback/link-local không.

    Chống SSRF: ngăn chặn redirect tới dải IP nội bộ như
    127.0.0.1, 169.254.169.254 (AWS metadata), 10.x.x.x, 192.168.x.x.

    Args:
        hostname: Tên miền hoặc IP cần kiểm tra.

    Returns:
        True nếu hostname là mục tiêu SSRF (cần chặn).
    """
    try:
        # Resolve DNS → IP
        addr_infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC)
    except (socket.gaierror, OSError):
        return False  # Không resolve được → không phải SSRF target

    for _family, _, _, _, sockaddr in addr_infos:
        ip_str = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            continue

        if addr.is_private or addr.is_loopback or addr.is_link_local:
            logger.warning(
                "⚠️ SSRF blocked: %s resolves to private IP %s",
                hostname,
                ip_str,
            )
            return True

    return False


def _safe_get(
    session: requests.Session,
    url: str,
    timeout: int = _REQUEST_TIMEOUT,
    **kwargs: Any,
) -> requests.Response:
    """HTTP GET với chống SSRF: manual redirect + IP validation.

    Không dùng allow_redirects=True (mặc định của requests)
    vì nó follow redirect mù quáng. Thay vào đó:
    1. Gửi request với ``allow_redirects=False``.
    2. Nếu 301/302/307/308 → kiểm tra Location hostname.
    3. Nếu hostname resolve tới private IP → raise.
    4. Lặp lại tối đa 5 hops.

    Args:
        session: requests.Session đã cấu hình.
        url: URL cần fetch.
        timeout: Timeout giây.
        **kwargs: Extra arguments cho session.get().

    Returns:
        Response cuối cùng.

    Raises:
        ConnectionError: Khi SSRF detected hoặc quá nhiều redirects.
    """
    # Kiểm tra URL gốc
    parsed = urlparse(url)
    if _is_ssrf_target(parsed.hostname or ""):
        raise ConnectionError(f"SSRF blocked: {url} resolves to private IP")

    current_url = url
    for _ in range(_MAX_REDIRECTS):
        response = session.get(
            current_url,
            timeout=timeout,
            allow_redirects=False,
            **kwargs,
        )

        # Không phải redirect → trả về luôn
        if response.status_code not in (301, 302, 303, 307, 308):
            return response

        # Lấy Location header
        location = response.headers.get("Location")
        if not location:
            return response

        # Resolve relative URL
        if location.startswith("/"):
            parsed_current = urlparse(current_url)
            location = f"{parsed_current.scheme}://{parsed_current.netloc}{location}"

        # Kiểm tra redirect target
        parsed_next = urlparse(location)
        next_host = parsed_next.hostname or ""

        # Chặn redirect tới non-HTTP schemes
        if parsed_next.scheme not in ("http", "https"):
            raise ConnectionError(
                f"SSRF blocked: redirect to non-HTTP scheme {parsed_next.scheme}://{next_host}"
            )

        if _is_ssrf_target(next_host):
            raise ConnectionError(
                f"SSRF blocked: redirect to private IP {next_host} (from {current_url})"
            )

        logger.debug(
            "  Redirect %d: %s → %s", response.status_code, current_url, location
        )
        current_url = location

    raise ConnectionError(f"Quá nhiều redirects (>{_MAX_REDIRECTS}) cho: {url}")


def _smart_truncate(text: str, max_tokens: int = _MAX_SUMMARY_TOKENS) -> str:
    """Cắt chuỗi thông minh dựa trên token budget.

    Sử dụng token_counter để ước lượng chính xác số tokens,
    cắt tại word boundary gần nhất.

    Args:
        text: Chuỗi cần cắt.
        max_tokens: Ngân sách token tối đa cho phép.

    Returns:
        Chuỗi đã cắt + "…" nếu vượt budget, giữ nguyên nếu không.
    """
    return truncate_to_token_budget(text, max_tokens)


# ── Module-level function for ProcessPool (pickling) ────────


def _trafilatura_extract(html: str) -> str | None:
    """Extract full-text từ HTML bằng trafilatura.

    Hàm module-level để có thể pickle → chạy trong ProcessPool.
    trafilatura.extract() là CPU-bound (phân tích cây HTML).

    Quality Gate: include_tables=True để không mất bảng số liệu.

    Args:
        html: HTML content cần extract.

    Returns:
        Text đã extract, hoặc None nếu thất bại.
    """
    try:
        text = trafilatura.extract(html, include_tables=True)
        if not text:
            return None
        # Quality Gate:
        # - Giữ nội dung dài bình thường.
        # - Với nội dung ngắn, chỉ drop khi thực sự nghèo thông tin.
        if len(text) < 150 and not _looks_like_quant_data(text):
            return None
        return text
    except Exception:  # noqa: BLE001
        return None


def _looks_like_quant_data(text: str) -> bool:
    """Heuristic phát hiện nội dung bảng/số liệu ngắn nhưng có giá trị."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    numeric_hits = len(_NUMERIC_TOKEN_RE.findall(text))
    table_markers = text.count("|") + text.count("\t")
    return numeric_hits >= 6 or (table_markers >= 4 and numeric_hits >= 2)


class RSSNewsFetcher(INewsFetcher):
    """Cào tin tức từ RSS feeds qua requests + feedparser.

    Implement INewsFetcher — chịu trách nhiệm:
    1. Fetch RSS feed qua HTTP với timeout + retry.
    2. Parse response bằng feedparser.
    3. Map từng entry sang domain entity ``Article``.
    4. Clean HTML bằng BeautifulSoup4 và smart truncate summary.
    5. Deep Scraper: Bóc tách full-text khi RSS summary quá ngắn.

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
        self._cpu_pool = ProcessPool(
            max_workers=_CPU_WORKERS,
            max_tasks=50,  # Recycle workers sau 50 tasks → tránh lxml memory leak
        )
        self._domain_breaker = DomainCircuitBreaker(
            failure_threshold=_CIRCUIT_BREAKER_THRESHOLD,
            open_ttl=_CIRCUIT_BREAKER_OPEN_TTL_SECONDS,
        )
        self._domain_delay_lock = threading.Lock()
        self._domain_next_allowed_at: dict[str, float] = {}
        self._fulltext_cache: Any | None = None
        try:
            from diskcache import Cache  # type: ignore[import-not-found]

            self._fulltext_cache = Cache(str(_FULL_TEXT_CACHE_DIR))
        except Exception as exc:  # noqa: BLE001
            logger.warning("DiskCache unavailable, deep cache disabled: %s", exc)

    def close(self) -> None:
        """Đóng HTTP session và CPU pool, giải phóng tài nguyên."""
        self._session.close()
        self._cpu_pool.stop()  # type: ignore[no-untyped-call]
        self._cpu_pool.join(timeout=5)
        if self._fulltext_cache is not None:
            try:
                self._fulltext_cache.close()
            except Exception:  # noqa: BLE001
                logger.debug("Không thể đóng diskcache fulltext.", exc_info=True)
        logger.debug("RSSNewsFetcher resources closed.")

    @property
    def supports_batch(self) -> bool:
        """RSSNewsFetcher hỗ trợ async batch qua httpx.AsyncClient."""
        return True

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
        logger.info(
            "Fetching RSS: %s (limit=%d, timeout=%ds)", url, limit, self._timeout
        )

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
        deep_scrape_budget = self._select_deep_scrape_links(entries)

        for entry in entries:
            article = self._parse_entry(
                entry,
                allow_deep_scrape=self._entry_link(entry) in deep_scrape_budget,
            )
            if article is not None:
                articles.append(article)

        logger.info(
            "Parsed %d/%d articles từ %s",
            len(articles),
            len(entries),
            url,
        )

        return articles

    def fetch_many(
        self,
        tasks: list[tuple[SourceCategory, SourceConfig]],
        limit: int = 10,
    ) -> dict[SourceCategory, list[Article]]:
        """Fetch batch nhiều RSS sources bằng async I/O an toàn."""
        if not tasks:
            return {}

        logger.info(
            "Batch crawling %d sources bằng async (max_connections=%d)",
            len(tasks),
            _ASYNC_MAX_CONNECTIONS,
        )

        try:
            return asyncio.run(self._fetch_many_async(tasks=tasks, limit=limit))
        except RuntimeError as exc:
            if "asyncio.run() cannot be called from a running event loop" not in str(
                exc
            ):
                raise

            # Fallback an toàn khi fetcher được gọi từ môi trường đã có event loop.
            result_holder: dict[str, dict[SourceCategory, list[Article]]] = {}
            error_holder: dict[str, Exception] = {}

            def _runner() -> None:
                try:
                    result_holder["result"] = asyncio.run(
                        self._fetch_many_async(tasks=tasks, limit=limit)
                    )
                except Exception as run_exc:  # noqa: BLE001
                    error_holder["error"] = run_exc

            thread = threading.Thread(target=_runner, daemon=True)
            thread.start()
            thread.join()

            if "error" in error_holder:
                raise RuntimeError(
                    "fetch_many async thread fallback thất bại"
                ) from error_holder["error"]
            return result_holder.get("result", {})

    async def _fetch_many_async(
        self,
        tasks: list[tuple[SourceCategory, SourceConfig]],
        limit: int,
    ) -> dict[SourceCategory, list[Article]]:
        """Async worker cho batch RSS crawling."""
        semaphore = asyncio.Semaphore(_ASYNC_MAX_CONNECTIONS)
        limits = httpx.Limits(
            max_connections=_ASYNC_MAX_CONNECTIONS,
            max_keepalive_connections=_ASYNC_MAX_CONNECTIONS,
        )
        timeout = httpx.Timeout(
            connect=_ASYNC_CONNECT_TIMEOUT,
            read=_ASYNC_READ_TIMEOUT,
            write=_ASYNC_WRITE_TIMEOUT,
            pool=_ASYNC_POOL_TIMEOUT,
        )

        result: dict[SourceCategory, list[Article]] = {}
        async with httpx.AsyncClient(
            headers={"User-Agent": _USER_AGENT},
            limits=limits,
            timeout=timeout,
            follow_redirects=False,
        ) as client:
            coros = [
                self._fetch_single_source_async(
                    client=client,
                    semaphore=semaphore,
                    category=category,
                    source=source,
                    limit=limit,
                )
                for category, source in tasks
            ]
            done = await asyncio.gather(*coros, return_exceptions=True)

        for item in done:
            if isinstance(item, Exception):
                logger.warning("Batch fetch source lỗi: %s", item)
                continue
            if not isinstance(item, tuple):
                logger.warning("Batch fetch source trả về kiểu không hợp lệ: %r", item)
                continue

            category, source_name, articles = item
            if category not in result:
                result[category] = []
            result[category].extend(articles)
            logger.info("  [%s] %s → %d bài", category, source_name, len(articles))

        return result

    async def _fetch_single_source_async(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        category: SourceCategory,
        source: SourceConfig,
        limit: int,
    ) -> tuple[SourceCategory, str, list[Article]]:
        """Fetch và parse một source trong batch async."""
        try:
            articles = await self._fetch_news_async(
                client=client,
                semaphore=semaphore,
                url=source.url,
                source_name=source.name,
                limit=limit,
            )
            return category, source.name, articles
        except Exception as exc:
            raise ConnectionError(f"{source.name}: {exc}") from exc

    async def _fetch_news_async(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        url: str,
        source_name: str,
        limit: int,
    ) -> list[Article]:
        """Async version của fetch_news cho batch mode."""
        raw_content = await self._http_get_async(
            client=client,
            semaphore=semaphore,
            url=url,
        )
        feed = feedparser.parse(raw_content)

        if feed.bozo and not feed.entries:
            exception_msg = str(getattr(feed, "bozo_exception", "Unknown error"))
            raise ConnectionError(
                f"Không thể parse RSS feed: {url}. Lỗi: {exception_msg}"
            )

        entries = feed.entries[:limit]
        deep_scrape_budget = self._select_deep_scrape_links(entries)
        coros = [
            self._parse_entry_async(
                entry=entry,
                source_name=source_name,
                client=client,
                semaphore=semaphore,
                allow_deep_scrape=self._entry_link(entry) in deep_scrape_budget,
            )
            for entry in entries
        ]
        parsed_entries = await asyncio.gather(*coros, return_exceptions=True)

        articles: list[Article] = []
        for item in parsed_entries:
            if isinstance(item, BaseException):
                logger.warning("  Parse entry lỗi (%s): %s", url, item)
                continue
            if item is not None:
                articles.append(item)

        logger.info("Parsed %d/%d articles từ %s", len(articles), len(entries), url)
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
            response = _safe_get(
                self._session,
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
                response = _safe_get(self._session, url, timeout=self._timeout)

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

    async def _http_get_async(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        url: str,
    ) -> bytes:
        """Async fetch URL với Conditional GET (ETag/Last-Modified)."""
        extra_headers = self._cache.get_headers(url)

        response = await self._safe_get_with_retries_async(
            client=client,
            semaphore=semaphore,
            url=url,
            headers=extra_headers,
        )

        if response.status_code == 304:
            cached = self._cache.get_cached_content(url)
            if cached is not None:
                logger.info("  304 Not Modified, dùng cache: %s", url)
                return cached
            logger.warning("  304 nhưng không có cache, fetch lại: %s", url)
            response = await self._safe_get_with_retries_async(
                client=client,
                semaphore=semaphore,
                url=url,
            )

        if response.status_code >= 400:
            raise ConnectionError(f"RSS feed trả về HTTP {response.status_code}: {url}")

        self._cache.update(
            url=url,
            content=response.content,
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
        )
        return response.content

    async def _safe_get_with_retries_async(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        url: str,
        headers: dict[str, str] | None = None,
        method: str = "GET",
    ) -> httpx.Response:
        """Async request với retry/backoff cho lỗi mạng và status retryable."""
        for attempt in range(_ASYNC_MAX_RETRIES + 1):
            try:
                response = await self._safe_get_async(
                    client=client,
                    semaphore=semaphore,
                    url=url,
                    headers=headers,
                    method=method,
                )
            except httpx.TimeoutException as exc:
                if attempt >= _ASYNC_MAX_RETRIES:
                    raise ConnectionError(
                        f"HTTP {method} timeout "
                        f"(connect={_ASYNC_CONNECT_TIMEOUT}s/"
                        f"read={_ASYNC_READ_TIMEOUT}s): {url}"
                    ) from exc
                await self._sleep_with_backoff(attempt=attempt)
                continue
            except httpx.RequestError as exc:
                if attempt >= _ASYNC_MAX_RETRIES:
                    raise ConnectionError(
                        f"Không thể kết nối tới URL ({method}): {url}. Lỗi: {exc}"
                    ) from exc
                await self._sleep_with_backoff(attempt=attempt)
                continue

            if (
                response.status_code in _ASYNC_RETRYABLE_STATUS_CODES
                and attempt < _ASYNC_MAX_RETRIES
            ):
                retry_after = self._parse_retry_after(
                    response.headers.get("Retry-After")
                )
                await self._sleep_with_backoff(
                    attempt=attempt,
                    retry_after=retry_after,
                )
                continue

            return response

        raise ConnectionError(f"Không thể fetch URL sau retry: {url}")

    async def _safe_get_async(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        url: str,
        headers: dict[str, str] | None = None,
        method: str = "GET",
    ) -> httpx.Response:
        """HTTP request async với manual redirect + SSRF validation."""
        parsed = urlparse(url)
        if _is_ssrf_target(parsed.hostname or ""):
            raise ConnectionError(f"SSRF blocked: {url} resolves to private IP")

        current_url = url
        for _ in range(_MAX_REDIRECTS):
            response = await self._throttled_request_async(
                client=client,
                semaphore=semaphore,
                url=current_url,
                headers=headers,
                method=method,
            )

            if response.status_code not in (301, 302, 303, 307, 308):
                return response

            location = response.headers.get("Location")
            if not location:
                return response

            next_url = urljoin(current_url, location)
            parsed_next = urlparse(next_url)
            next_host = parsed_next.hostname or ""
            if parsed_next.scheme not in ("http", "https"):
                raise ConnectionError(
                    f"SSRF blocked: redirect to non-HTTP scheme {parsed_next.scheme}://{next_host}"
                )
            if _is_ssrf_target(next_host):
                raise ConnectionError(
                    "SSRF blocked: redirect to private IP "
                    f"{next_host} (from {current_url})"
                )

            logger.debug("  Async redirect %s → %s", current_url, next_url)
            current_url = next_url

        raise ConnectionError(f"Quá nhiều redirects (>{_MAX_REDIRECTS}) cho: {url}")

    async def _throttled_request_async(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        url: str,
        headers: dict[str, str] | None = None,
        method: str = "GET",
    ) -> httpx.Response:
        """HTTP request bọc semaphore + jitter delay để giảm burst."""
        await self._wait_for_domain_jitter(url)
        async with semaphore:
            return await client.request(
                method,
                url,
                headers=headers,
                follow_redirects=False,
            )

    async def _sleep_with_backoff(self, attempt: int, retry_after: float = 0.0) -> None:
        """Sleep retry theo backoff 2/4/8 giây, tôn trọng Retry-After."""
        backoff = float(2 ** (attempt + 1))
        delay = max(backoff, retry_after)
        await asyncio.sleep(delay)

    async def _wait_for_domain_jitter(self, url: str) -> None:
        """Áp dụng giãn cách request theo domain để tránh bot-detection burst."""
        host = (urlparse(url).hostname or "").lower()
        if not host:
            await asyncio.sleep(random.uniform(_ASYNC_JITTER_MIN, _ASYNC_JITTER_MAX))
            return

        delay = random.uniform(_ASYNC_JITTER_MIN, _ASYNC_JITTER_MAX)
        now = time.monotonic()

        with self._domain_delay_lock:
            next_allowed = self._domain_next_allowed_at.get(host, now)
            scheduled_at = max(now, next_allowed)
            self._domain_next_allowed_at[host] = scheduled_at + delay

        wait_for = max(0.0, scheduled_at - now)
        if wait_for > 0:
            await asyncio.sleep(wait_for)

    @staticmethod
    def _parse_retry_after(value: str | None) -> float:
        """Parse Retry-After header thành giây."""
        if not value:
            return 0.0
        try:
            return max(0.0, float(value.strip()))
        except ValueError:
            return 0.0

    # ── Private: Deep Scraper (Full-text Extraction) ────────

    def _fetch_full_text(self, url: str) -> str | None:
        """Bóc tách full-text từ URL gốc bằng trafilatura.

        Tách thành 2 phase để tránh GIL bottleneck:
        1. I/O phase: HTTP download (thread-safe).
        2. CPU phase: trafilatura extract (ProcessPool).

        Defensive Programming:
        - Bắt mọi exception HTTP (Timeout, 403, ConnectionError).
        - Trả về None nếu lỗi → pipeline vẫn dùng RSS summary cũ.
        - Không làm crash toàn bộ mẻ cào tin.

        Args:
            url: URL gốc của bài báo.

        Returns:
            Full-text đã extract, hoặc None nếu thất bại.
        """
        cached = self._get_cached_full_text(url)
        if cached is not None:
            logger.debug("  Deep cache hit: %s", url)
            return cached

        # ── Phase 1: I/O-bound (thread-safe) ──
        try:
            response = _safe_get(self._session, url, timeout=self._timeout)
            response.raise_for_status()
        except requests.Timeout:
            logger.warning("  Deep Scraper timeout sau %ds: %s", self._timeout, url)
            return None
        except requests.ConnectionError:
            logger.warning("  Deep Scraper không kết nối được: %s", url)
            return None
        except requests.HTTPError as exc:
            logger.warning(
                "  Deep Scraper bị chặn (HTTP %s): %s",
                exc.response.status_code if exc.response else "?",
                url,
            )
            return None
        except requests.RequestException as exc:
            logger.warning("  Deep Scraper lỗi không xác định: %s — %s", url, exc)
            return None

        text = self._extract_full_text_with_pool(response.text, url)
        if text is not None:
            self._set_cached_full_text(url, text)
        return text

    async def _fetch_full_text_async(
        self,
        url: str,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
    ) -> str | None:
        """Async deep scraper với fallback curl_cffi khi gặp HTTP 403."""
        cached = await asyncio.to_thread(self._get_cached_full_text, url)
        if cached is not None:
            logger.debug("  Deep cache hit: %s", url)
            return cached

        html_text: str | None = None
        try:
            response = await self._safe_get_with_retries_async(
                client=client,
                semaphore=semaphore,
                url=url,
            )
        except ConnectionError as exc:
            logger.warning("  Deep Scraper không kết nối được: %s (%s)", url, exc)
            return None

        if response.status_code == 403:
            logger.info(
                "  Deep Scraper gặp HTTP 403, thử fallback curl_cffi: %s",
                url,
            )
            html_text = await self._fetch_html_with_curl_cffi_async(
                url=url,
                semaphore=semaphore,
            )
            if html_text is None:
                logger.warning("  Fallback curl_cffi thất bại: %s", url)
                return None
        elif response.status_code >= 400:
            logger.warning(
                "  Deep Scraper bị chặn (HTTP %d): %s",
                response.status_code,
                url,
            )
            return None
        else:
            html_text = response.text

        text = await asyncio.to_thread(
            self._extract_full_text_with_pool,
            html_text,
            url,
        )
        if text is not None:
            await asyncio.to_thread(self._set_cached_full_text, url, text)
        return text

    def _fetch_full_text_half_open_probe(self, url: str) -> str | None:
        """Probe deep scrape khi breaker ở HALF_OPEN: ưu tiên curl_cffi."""
        cached = self._get_cached_full_text(url)
        if cached is not None:
            return cached

        html_text = self._fetch_html_with_curl_cffi(url)
        if not html_text:
            return None

        text = self._extract_full_text_with_pool(html_text, url)
        if text is not None:
            self._set_cached_full_text(url, text)
        return text

    async def _fetch_full_text_half_open_probe_async(
        self,
        url: str,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
    ) -> str | None:
        """Probe deep scrape khi breaker ở HALF_OPEN: ưu tiên curl_cffi."""
        cached = await asyncio.to_thread(self._get_cached_full_text, url)
        if cached is not None:
            return cached

        html_text = await self._fetch_html_with_curl_cffi_async(
            url=url,
            semaphore=semaphore,
        )
        if not html_text:
            return None

        text = await asyncio.to_thread(
            self._extract_full_text_with_pool,
            html_text,
            url,
        )
        if text is not None:
            await asyncio.to_thread(self._set_cached_full_text, url, text)
        return text

    async def _fetch_html_with_curl_cffi_async(
        self,
        url: str,
        semaphore: asyncio.Semaphore,
    ) -> str | None:
        """Run curl_cffi fallback trong thread để không chặn event loop."""
        await self._wait_for_domain_jitter(url)
        async with semaphore:
            return await asyncio.to_thread(self._fetch_html_with_curl_cffi, url)

    def _fetch_html_with_curl_cffi(self, url: str) -> str | None:
        """Fallback dùng curl_cffi giả lập fingerprint trình duyệt."""
        global curl_cffi_requests

        parsed = urlparse(url)
        if _is_ssrf_target(parsed.hostname or ""):
            logger.warning("  curl_cffi blocked bởi SSRF policy: %s", url)
            return None

        if curl_cffi_requests is None:
            try:
                curl_cffi_requests = importlib.import_module("curl_cffi.requests")
            except Exception as exc:  # noqa: BLE001
                logger.debug("  curl_cffi chưa khả dụng: %s", exc)
                return None

        try:
            response = curl_cffi_requests.get(
                url,
                timeout=self._timeout,
                impersonate="chrome120",
            )
            response.raise_for_status()
            return str(response.text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  curl_cffi fallback lỗi: %s — %s", url, exc)
            return None

    def _extract_full_text_with_pool(self, html_text: str, url: str) -> str | None:
        """Extract full-text bằng ProcessPool để tránh block GIL."""
        try:
            future = self._cpu_pool.schedule(
                _trafilatura_extract,
                args=[html_text],
                timeout=30,
            )
            text = future.result()
        except FutureTimeoutError:
            future.cancel()  # type: ignore[no-untyped-call]
            logger.warning("  trafilatura timeout 30s (worker killed): %s", url)
            return None
        except Exception:  # noqa: BLE001
            logger.warning("  trafilatura extract thất bại: %s", url)
            return None

        if not isinstance(text, str) or not text.strip():
            logger.debug("  trafilatura trả về empty cho: %s", url)
            return None

        logger.info("  ✅ Deep Scraper: extracted %d chars từ %s", len(text), url)
        return text.strip()

    def _get_cached_full_text(self, url: str) -> str | None:
        """Đọc full-text cache từ diskcache nếu có."""
        if self._fulltext_cache is None:
            return None
        cache_key = self._canonical_cache_key(url)
        try:
            cached = self._fulltext_cache.get(cache_key)
        except Exception:  # noqa: BLE001
            logger.debug("  Lỗi đọc diskcache fulltext: %s", cache_key, exc_info=True)
            return None
        if isinstance(cached, str) and cached.strip():
            return cached.strip()
        return None

    def _set_cached_full_text(self, url: str, text: str) -> None:
        """Ghi full-text vào diskcache để tái sử dụng cho lần chạy sau."""
        if self._fulltext_cache is None or not text.strip():
            return
        cache_key = self._canonical_cache_key(url)
        try:
            self._fulltext_cache.set(cache_key, text.strip())
        except Exception:  # noqa: BLE001
            logger.debug("  Lỗi ghi diskcache fulltext: %s", cache_key, exc_info=True)

    @staticmethod
    def _canonical_cache_key(url: str) -> str:
        """Tạo cache key ổn định cho full-text."""
        return canonicalize_url(url)

    def _get_domain_breaker_state(self, domain: str) -> BreakerState:
        return self._domain_breaker.get_state(domain)

    def _record_domain_success(self, domain: str) -> None:
        self._domain_breaker.record_success(domain)

    def _record_domain_failure(self, domain: str) -> BreakerState:
        return self._domain_breaker.record_failure(domain)

    @staticmethod
    def _entry_link(entry: Any) -> str:
        return str(entry.get("link", "")).strip()

    def _select_deep_scrape_links(self, entries: list[Any]) -> set[str]:
        """Chọn link được phép deep scrape theo semantic+recency budgeting."""
        if not entries:
            return set()

        now_utc = datetime.now(tz=UTC)
        candidates: list[tuple[int, datetime, str]] = []
        for entry in entries:
            title = str(entry.get("title", "")).strip()
            link = self._entry_link(entry)
            if not title or not link:
                continue

            summary_raw = entry.get("summary", "") or entry.get("description", "") or ""
            summary = self._clean_html(summary_raw)
            score = score_relevancy(title, summary)
            published_date = self._parse_date(entry)
            if published_date.tzinfo is None:
                published_date = published_date.replace(tzinfo=UTC)
            age_seconds = max(
                0.0,
                (now_utc - published_date.astimezone(UTC)).total_seconds(),
            )

            # Recency bonus: ưu tiên bài mới trong 72h gần nhất.
            if age_seconds <= 6 * 3600:
                score += 20
            elif age_seconds <= 24 * 3600:
                score += 12
            elif age_seconds <= 72 * 3600:
                score += 6

            if len(summary) < _MIN_SUMMARY_THRESHOLD:
                score += 5

            candidates.append((score, published_date, link))

        if not candidates:
            return set()

        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        budget = min(_DEEP_SCRAPE_MAX_PER_FEED, len(candidates))
        selected = {
            link for score, _, link in candidates[:budget] if score > 0 or budget <= 2
        }

        if not selected:
            # fallback tối thiểu: giữ 2 bài mới nhất để không bỏ sót coverage.
            selected = {link for _, _, link in candidates[:2]}

        logger.debug(
            "  Deep scrape budget selected %d/%d links",
            len(selected),
            len(candidates),
        )
        return selected

    def _resolve_google_news_url(self, url: str) -> str:
        """Resolve Google News RSS redirect sang URL gốc (sync)."""
        if not is_google_news_url(url):
            return url

        try:
            response = _safe_get(
                self._session,
                url,
                timeout=min(self._timeout, _GOOGLE_RESOLVE_TIMEOUT_SECONDS),
                stream=True,
            )
            resolved = str(response.url or url)
            response.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug("  Resolve Google News thất bại: %s (%s)", url, exc)
            return url

        if not resolved:
            return url
        parsed = urlparse(resolved)
        host = parsed.hostname or ""
        if parsed.scheme in ("http", "https") and not _is_ssrf_target(host):
            logger.debug("  Google redirect resolved: %s -> %s", url, resolved)
            return resolved
        return url

    async def _resolve_google_news_url_async(
        self,
        url: str,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
    ) -> str:
        """Resolve Google News RSS redirect sang URL gốc (async)."""
        if not is_google_news_url(url):
            return url

        try:
            response = await self._safe_get_with_retries_async(
                client=client,
                semaphore=semaphore,
                url=url,
                method="HEAD",
            )
            resolved = str(response.url or url)
            if response.status_code in (405, 501) or is_google_news_url(resolved):
                response = await self._safe_get_with_retries_async(
                    client=client,
                    semaphore=semaphore,
                    url=url,
                    method="GET",
                )
                resolved = str(response.url or url)
        except Exception as exc:  # noqa: BLE001
            logger.debug("  Resolve Google News async thất bại: %s (%s)", url, exc)
            return url

        if not resolved:
            return url
        parsed = urlparse(resolved)
        host = parsed.hostname or ""
        if parsed.scheme in ("http", "https") and not _is_ssrf_target(host):
            logger.debug("  Google redirect resolved async: %s -> %s", url, resolved)
            return resolved
        return url

    # ── Private: Entry Parsing ──────────────────────────────

    def _parse_entry(
        self,
        entry: Any,
        source_name: str | None = None,
        allow_deep_scrape: bool = True,
    ) -> Article | None:
        """Parse một RSS entry thành Article.

        Nếu RSS summary quá ngắn (< 500 ký tự), tự động kích hoạt
        Deep Scraper để bóc tách full-text từ URL gốc.

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

        resolved_link = self._resolve_google_news_url(link)
        summary_raw = entry.get("summary", "") or entry.get("description", "") or ""
        summary = self._clean_html(summary_raw)
        selected_source_name = source_name or self._source_name

        if (
            allow_deep_scrape
            and len(summary) < _MIN_SUMMARY_THRESHOLD
            and is_allowed_by_policy(resolved_link)
        ):
            domain = (urlparse(resolved_link).hostname or "").lower()
            state = self._get_domain_breaker_state(domain)

            if state == BreakerState.OPEN:
                remaining = self._domain_breaker.get_open_remaining(domain)
                logger.info(
                    "  ⚡ Circuit breaker OPEN cho %s (%.0fs còn lại), bỏ qua: %s",
                    domain,
                    remaining,
                    title[:60],
                )
            else:
                logger.info(
                    "  Teaser detected (%d chars < %d): %s",
                    len(summary),
                    _MIN_SUMMARY_THRESHOLD,
                    title[:60],
                )
                if state == BreakerState.HALF_OPEN:
                    logger.info("  HALF_OPEN probe bằng curl_cffi cho %s", domain)
                    full_text = self._fetch_full_text_half_open_probe(resolved_link)
                else:
                    full_text = self._fetch_full_text(resolved_link)

                if full_text:
                    summary = full_text
                    self._record_domain_success(domain)
                else:
                    next_state = self._record_domain_failure(domain)
                    if next_state == BreakerState.OPEN:
                        logger.warning(
                            "  ⚡ Circuit breaker TRIPPED cho %s "
                            "(>= %d failures liên tiếp)",
                            domain,
                            _CIRCUIT_BREAKER_THRESHOLD,
                        )

        # Token-based truncation (thay vì char-based)
        summary = _smart_truncate(summary, _MAX_SUMMARY_TOKENS)

        published_date = self._parse_date(entry)

        return Article(
            title=title,
            summary=summary,
            source_name=selected_source_name,
            published_date=published_date,
            url=resolved_link,
        )

    async def _parse_entry_async(
        self,
        entry: Any,
        source_name: str,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        allow_deep_scrape: bool = True,
    ) -> Article | None:
        """Parse một RSS entry trong async batch mode."""
        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()

        if not title or not link:
            logger.warning(
                "Bỏ qua entry thiếu title/link: title=%r, link=%r",
                title,
                link,
            )
            return None

        resolved_link = await self._resolve_google_news_url_async(
            url=link,
            client=client,
            semaphore=semaphore,
        )
        summary_raw = entry.get("summary", "") or entry.get("description", "") or ""
        summary = self._clean_html(summary_raw)

        if (
            allow_deep_scrape
            and len(summary) < _MIN_SUMMARY_THRESHOLD
            and is_allowed_by_policy(resolved_link)
        ):
            domain = (urlparse(resolved_link).hostname or "").lower()
            state = self._get_domain_breaker_state(domain)
            if state == BreakerState.OPEN:
                remaining = self._domain_breaker.get_open_remaining(domain)
                logger.info(
                    "  ⚡ Circuit breaker OPEN cho %s (%.0fs còn lại), bỏ qua: %s",
                    domain,
                    remaining,
                    title[:60],
                )
            else:
                logger.info(
                    "  Teaser detected (%d chars < %d): %s",
                    len(summary),
                    _MIN_SUMMARY_THRESHOLD,
                    title[:60],
                )
                if state == BreakerState.HALF_OPEN:
                    logger.info("  HALF_OPEN probe bằng curl_cffi cho %s", domain)
                    full_text = await self._fetch_full_text_half_open_probe_async(
                        url=resolved_link,
                        client=client,
                        semaphore=semaphore,
                    )
                else:
                    full_text = await self._fetch_full_text_async(
                        url=resolved_link,
                        client=client,
                        semaphore=semaphore,
                    )
                if full_text:
                    summary = full_text
                    self._record_domain_success(domain)
                else:
                    next_state = self._record_domain_failure(domain)
                    if next_state == BreakerState.OPEN:
                        logger.warning(
                            "  ⚡ Circuit breaker TRIPPED cho %s "
                            "(>= %d failures liên tiếp)",
                            domain,
                            _CIRCUIT_BREAKER_THRESHOLD,
                        )

        summary = _smart_truncate(summary, _MAX_SUMMARY_TOKENS)
        published_date = self._parse_date(entry)
        return Article(
            title=title,
            summary=summary,
            source_name=source_name,
            published_date=published_date,
            url=resolved_link,
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
                return datetime.fromtimestamp(timestamp, tz=UTC)
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
                return datetime.fromtimestamp(timestamp, tz=UTC)
            except (TypeError, ValueError, OverflowError):
                pass

        logger.debug("Không parse được date, dùng UTC now.")
        return datetime.now(tz=UTC)
