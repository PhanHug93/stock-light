"""RSS ETag Cache — lưu trữ ETag/Last-Modified cho Conditional GET.

Giúp tiết kiệm băng thông bằng cách gửi ``If-None-Match`` và
``If-Modified-Since`` headers. Nếu RSS feed chưa thay đổi,
server trả ``304 Not Modified`` → bỏ qua, không tải lại toàn bộ XML.

Cache lưu tại ``.cache/rss_etags.json`` trong project root.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(".cache")
_CACHE_FILENAME = "rss_etags.json"


class RSSCache:
    """Quản lý ETag/Last-Modified cache cho RSS feeds.

    Mỗi URL lưu 1 entry::

        {
            "https://example.com/rss": {
                "etag": "\"abc123\"",
                "last_modified": "Thu, 01 Jan 2026 00:00:00 GMT",
                "content": "<rss>...</rss>"
            }
        }

    Args:
        cache_dir: Thư mục lưu cache (default: ``.cache/``).
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._cache_path = self._cache_dir / _CACHE_FILENAME
        self._data: dict[str, dict[str, str]] = self._load()

    def get_headers(self, url: str) -> dict[str, str]:
        """Lấy conditional headers cho một URL.

        Args:
            url: URL của RSS feed.

        Returns:
            Dict chứa If-None-Match / If-Modified-Since headers.
            Rỗng nếu chưa có cache cho URL này.
        """
        entry = self._data.get(url)
        if entry is None:
            return {}

        headers: dict[str, str] = {}
        if entry.get("etag"):
            headers["If-None-Match"] = entry["etag"]
        if entry.get("last_modified"):
            headers["If-Modified-Since"] = entry["last_modified"]

        return headers

    def get_cached_content(self, url: str) -> bytes | None:
        """Lấy nội dung RSS đã cache cho URL.

        Args:
            url: URL của RSS feed.

        Returns:
            Bytes content nếu có cache, None nếu chưa cache.
        """
        entry = self._data.get(url)
        if entry is None:
            return None

        content = entry.get("content")
        if content:
            return content.encode("utf-8")

        return None

    def update(
        self,
        url: str,
        content: bytes,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> None:
        """Cập nhật cache cho URL sau khi fetch thành công (200 OK).

        Args:
            url: URL vừa fetch.
            content: Raw response content.
            etag: ETag header từ response (nếu có).
            last_modified: Last-Modified header từ response (nếu có).
        """
        self._data[url] = {
            "etag": etag or "",
            "last_modified": last_modified or "",
            "content": content.decode("utf-8", errors="replace"),
        }
        self._save()

    # ── Private ──────────────────────────────────────────────

    def _load(self) -> dict[str, dict[str, str]]:
        """Load cache từ file JSON."""
        if not self._cache_path.exists():
            return {}

        try:
            raw = self._cache_path.read_text(encoding="utf-8")
            data: Any = json.loads(raw)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Lỗi đọc RSS cache, bỏ qua: %s", exc)

        return {}

    def _save(self) -> None:
        """Lưu cache ra file JSON."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Lỗi lưu RSS cache: %s", exc)
