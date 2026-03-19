"""RSS News Fetcher — implementation of INewsFetcher.

Cào tin tức từ RSS feeds sử dụng ``requests`` + ``feedparser``.

Bảo mật & chịu tải:
    - Timeout 15s cho mỗi HTTP request.
    - Retry 2 lần với exponential backoff khi gặp 5xx.
    - User-Agent header để tránh bị RSS servers block.
    - HTML cleaning bằng BeautifulSoup4 (an toàn, xử lý nested/malformed HTML).
    - Token-based truncation thay vì character count.
    - Deep Scraper: Tự động bóc tách full-text khi RSS summary quá ngắn.
    - ProcessPool cho trafilatura (CPU-bound) tránh GIL blocking.
"""

from __future__ import annotations

import calendar
import ipaddress
import logging
import re
import socket
import threading
from collections import defaultdict
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

import feedparser  # type: ignore[import-untyped]
import requests
import trafilatura
from bs4 import BeautifulSoup
from pebble import ProcessPool
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from chahi.core.entities import Article
from chahi.core.interfaces import INewsFetcher
from chahi.infrastructure.llm.token_counter import truncate_to_token_budget
from chahi.infrastructure.rss.rss_cache import RSSCache

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")

_REQUEST_TIMEOUT: int = 15  # seconds mỗi request
_MAX_SUMMARY_TOKENS: int = 1500  # Token budget per article summary
_MIN_SUMMARY_THRESHOLD: int = 500  # Phase 10: ngưỡng kích hoạt Deep Scraper
_USER_AGENT: str = "ChaHi/0.3.0 RSS Fetcher (+https://github.com/chahi)"
_MAX_RETRIES: int = 2  # retry cho 5xx errors
_CPU_WORKERS: int = 2  # ProcessPool workers cho trafilatura
_CIRCUIT_BREAKER_THRESHOLD: int = 2  # Sau N failures liên tiếp → ngắt domain
_MAX_REDIRECTS: int = 5  # Tối đa redirect hops (chống SSRF)


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

    Args:
        html: HTML content cần extract.

    Returns:
        Text đã extract, hoặc None nếu thất bại.
    """
    try:
        return trafilatura.extract(html)
    except Exception:  # noqa: BLE001
        return None


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
        # Circuit breaker: domain → số failures liên tiếp (thread-safe)
        self._domain_failures: dict[str, int] = defaultdict(int)
        self._breaker_lock = threading.Lock()

    def close(self) -> None:
        """Đóng HTTP session và CPU pool, giải phóng tài nguyên."""
        self._session.close()
        self._cpu_pool.stop()
        self._cpu_pool.join(timeout=5)
        logger.debug("RSSNewsFetcher resources closed.")

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

        # ── Phase 2: CPU-bound (ProcessPool, tránh GIL) ──
        # Dùng pebble.ProcessPool — cancel() sends SIGKILL nếu timeout,
        # ngăn zombie workers exhaustion.
        try:
            future = self._cpu_pool.schedule(
                _trafilatura_extract,
                args=(response.text,),
                timeout=30,
            )
            text = future.result()
        except FutureTimeoutError:
            future.cancel()
            logger.warning("  trafilatura timeout 30s (worker killed): %s", url)
            return None
        except Exception:  # noqa: BLE001
            logger.warning("  trafilatura extract thất bại: %s", url)
            return None

        if not text or not text.strip():
            logger.debug("  trafilatura trả về empty cho: %s", url)
            return None

        logger.info(
            "  ✅ Deep Scraper: extracted %d chars từ %s",
            len(text),
            url,
        )
        return text.strip()

    # ── Private: Entry Parsing ──────────────────────────────

    def _parse_entry(self, entry: Any) -> Article | None:
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

        summary_raw = entry.get("summary", "") or entry.get("description", "") or ""
        summary = self._clean_html(summary_raw)

        # ── Phase 10: Deep Scraper (with Thread-safe Circuit Breaker) ──
        # Nếu RSS summary quá ngắn → bóc tách full-text từ URL gốc
        if len(summary) < _MIN_SUMMARY_THRESHOLD:
            domain = urlparse(link).netloc

            with self._breaker_lock:
                failures = self._domain_failures[domain]

            if failures >= _CIRCUIT_BREAKER_THRESHOLD:
                logger.info(
                    "  ⚡ Circuit breaker OPEN cho %s (%d failures), bỏ qua: %s",
                    domain,
                    failures,
                    title[:60],
                )
            else:
                logger.info(
                    "  Teaser detected (%d chars < %d): %s",
                    len(summary),
                    _MIN_SUMMARY_THRESHOLD,
                    title[:60],
                )
                full_text = self._fetch_full_text(link)
                with self._breaker_lock:
                    if full_text:
                        summary = full_text
                        self._domain_failures[domain] = 0
                    else:
                        self._domain_failures[domain] += 1
                        if self._domain_failures[domain] >= _CIRCUIT_BREAKER_THRESHOLD:
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
