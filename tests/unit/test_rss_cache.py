"""Unit tests cho RSS ETag/Last-Modified caching."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chahi.infrastructure.rss.rss_cache import RSSCache

if TYPE_CHECKING:
    from pathlib import Path


class TestRSSCache:
    """Tests cho RSSCache."""

    def test_empty_cache_returns_empty_headers(self, tmp_path: Path) -> None:
        """Chưa có cache → headers rỗng."""
        cache = RSSCache(cache_dir=tmp_path)
        headers = cache.get_headers("https://example.com/rss")
        assert headers == {}

    def test_empty_cache_returns_none_content(self, tmp_path: Path) -> None:
        """Chưa có cache → content None."""
        cache = RSSCache(cache_dir=tmp_path)
        assert cache.get_cached_content("https://example.com/rss") is None

    def test_update_and_get_headers(self, tmp_path: Path) -> None:
        """Sau update, get_headers phải trả conditional headers."""
        cache = RSSCache(cache_dir=tmp_path)
        cache.update(
            url="https://example.com/rss",
            content=b"<rss>data</rss>",
            etag='"abc123"',
            last_modified="Thu, 01 Jan 2026 00:00:00 GMT",
        )

        headers = cache.get_headers("https://example.com/rss")
        assert headers["If-None-Match"] == '"abc123"'
        assert headers["If-Modified-Since"] == "Thu, 01 Jan 2026 00:00:00 GMT"

    def test_update_and_get_content(self, tmp_path: Path) -> None:
        """Sau update, get_cached_content phải trả bytes."""
        cache = RSSCache(cache_dir=tmp_path)
        cache.update(
            url="https://example.com/rss",
            content=b"<rss>data</rss>",
        )

        content = cache.get_cached_content("https://example.com/rss")
        assert content == b"<rss>data</rss>"

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        """Cache phải persist qua file, instance mới đọc lại được."""
        cache1 = RSSCache(cache_dir=tmp_path)
        cache1.update(
            url="https://example.com/rss",
            content=b"<rss>persisted</rss>",
            etag='"xyz"',
        )

        # Instance mới load từ file
        cache2 = RSSCache(cache_dir=tmp_path)
        headers = cache2.get_headers("https://example.com/rss")
        assert headers["If-None-Match"] == '"xyz"'

        content = cache2.get_cached_content("https://example.com/rss")
        assert content == b"<rss>persisted</rss>"

    def test_no_etag_only_last_modified(self, tmp_path: Path) -> None:
        """Headers chỉ trả If-Modified-Since nếu không có ETag."""
        cache = RSSCache(cache_dir=tmp_path)
        cache.update(
            url="https://example.com/rss",
            content=b"data",
            last_modified="Mon, 19 Mar 2026 00:00:00 GMT",
        )

        headers = cache.get_headers("https://example.com/rss")
        assert "If-None-Match" not in headers
        assert headers["If-Modified-Since"] == "Mon, 19 Mar 2026 00:00:00 GMT"

    def test_corrupted_cache_file(self, tmp_path: Path) -> None:
        """File cache bị hỏng → load rỗng, không crash."""
        cache_file = tmp_path / "rss_etags.json"
        cache_file.write_text("not json!!!", encoding="utf-8")

        cache = RSSCache(cache_dir=tmp_path)
        assert cache.get_headers("https://example.com") == {}

    def test_multiple_urls(self, tmp_path: Path) -> None:
        """Cache lưu riêng biệt cho mỗi URL."""
        cache = RSSCache(cache_dir=tmp_path)
        cache.update("https://a.com/rss", b"data-a", etag='"a"')
        cache.update("https://b.com/rss", b"data-b", etag='"b"')

        assert cache.get_headers("https://a.com/rss")["If-None-Match"] == '"a"'
        assert cache.get_headers("https://b.com/rss")["If-None-Match"] == '"b"'
