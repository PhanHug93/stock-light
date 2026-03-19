"""RSS ETag Cache — lưu trữ ETag/Last-Modified cho Conditional GET.

Giúp tiết kiệm băng thông bằng cách gửi ``If-None-Match`` và
``If-Modified-Since`` headers. Nếu RSS feed chưa thay đổi,
server trả ``304 Not Modified`` → bỏ qua, không tải lại toàn bộ XML.

Chống OOM (Split Storage):
    - Metadata (ETag, Last-Modified, content_hash) lưu trong
      ``rss_etags.json`` (vài KB, nạp hết vào RAM).
    - Raw content lưu file riêng lẻ trong ``.cache/content/``
      theo SHA-256 hash. Chỉ đọc khi cache hit (304).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(".cache")
_METADATA_FILENAME = "rss_etags.json"
_CONTENT_DIR = "content"


class RSSCache:
    """Quản lý ETag/Last-Modified cache cho RSS feeds.

    Split Storage Pattern:
        Metadata JSON (nhẹ, nạp hết vào RAM)::

            {
                "https://example.com/rss": {
                    "etag": "\\"abc123\\"",
                    "last_modified": "Thu, 01 Jan 2026 00:00:00 GMT",
                    "content_hash": "a1b2c3d4e5f6..."
                }
            }

        Content files (nặng, chỉ đọc khi cache hit)::

            .cache/content/a1b2c3d4e5f6...dat

    Args:
        cache_dir: Thư mục lưu cache (default: ``.cache/``).
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._metadata_path = self._cache_dir / _METADATA_FILENAME
        self._content_dir = self._cache_dir / _CONTENT_DIR
        self._data: dict[str, dict[str, str]] = self._load_metadata()

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
        """Lấy nội dung RSS đã cache cho URL (lazy-load từ disk).

        Chỉ đọc file content khi thực sự cần (304 cache hit).
        Không nạp content vào RAM khi khởi tạo.

        Args:
            url: URL của RSS feed.

        Returns:
            Bytes content nếu có cache, None nếu chưa cache.
        """
        entry = self._data.get(url)
        if entry is None:
            return None

        content_hash = entry.get("content_hash")
        if not content_hash:
            # Backward compat: old format có "content" inline
            content = entry.get("content")
            if content:
                return content.encode("utf-8")
            return None

        # Đọc từ file riêng lẻ
        content_path = self._content_dir / f"{content_hash}.dat"
        if not content_path.exists():
            logger.warning("Cache content file mất: %s", content_path)
            return None

        try:
            return content_path.read_bytes()
        except OSError as exc:
            logger.warning("Lỗi đọc cache content: %s", exc)
            return None

    def update(
        self,
        url: str,
        content: bytes,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> None:
        """Cập nhật cache cho URL sau khi fetch thành công (200 OK).

        Lưu content ra file riêng (SHA-256 hash), metadata vào JSON.

        Args:
            url: URL vừa fetch.
            content: Raw response content.
            etag: ETag header từ response (nếu có).
            last_modified: Last-Modified header từ response (nếu có).
        """
        # Hash content → tên file
        content_hash = hashlib.sha256(content).hexdigest()[:16]

        # Lưu content ra file riêng
        self._save_content(content_hash, content)

        # Cập nhật metadata (KHÔNG lưu content trong JSON)
        self._data[url] = {
            "etag": etag or "",
            "last_modified": last_modified or "",
            "content_hash": content_hash,
        }
        self._save_metadata()

    # ── Private ──────────────────────────────────────────────

    def _load_metadata(self) -> dict[str, dict[str, str]]:
        """Load metadata từ file JSON (nhẹ, chỉ vài KB)."""
        if not self._metadata_path.exists():
            return {}

        try:
            raw = self._metadata_path.read_text(encoding="utf-8")
            data: Any = json.loads(raw)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Lỗi đọc RSS cache metadata, bỏ qua: %s", exc)

        return {}

    def _save_metadata(self) -> None:
        """Lưu metadata ra file JSON."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._metadata_path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Lỗi lưu RSS cache metadata: %s", exc)

    def _save_content(self, content_hash: str, content: bytes) -> None:
        """Lưu content ra file riêng trong .cache/content/."""
        try:
            self._content_dir.mkdir(parents=True, exist_ok=True)
            content_path = self._content_dir / f"{content_hash}.dat"
            content_path.write_bytes(content)
        except OSError as exc:
            logger.warning("Lỗi lưu cache content: %s", exc)
