"""File Memory Manager — fallback implementation of IMemoryManager.

Lưu nhận định phân tích ra file ``last_context.md`` trên đĩa.
Hoạt động ngay lập tức, không cần server hay dependency bên ngoài.

Dùng làm fallback khi chưa cắm MCP server thật.

An toàn concurrent (TOCTOU-safe):
    Sử dụng ``filelock`` với merge-on-write pattern:
    - Acquire lock → đọc file hiện tại → merge → ghi → release lock.
    - Tránh TOCTOU: toàn bộ read-modify-write là atomic trong lock.
    - Nhiều profile ghi đồng thời sẽ merge chứ không đè nhau.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from filelock import FileLock, Timeout

from chahi.core.interfaces import IMemoryManager

logger = logging.getLogger(__name__)

_LOCK_TIMEOUT: int = 10  # seconds chờ acquire lock
# Invisible boundary: LLM không bao giờ tự sinh ra chuỗi này
# (không dùng Markdown --- vì LLM hay tự đặt --- trong báo cáo)
_SECTION_SEPARATOR = "\n<!-- CHAHI_SECTION_f7a3b9e1 -->\n"
_SECTION_HEADER_RE = re.compile(r"<!-- \[(.+?)\] Saved: .+ -->")


class FileMemoryManager(IMemoryManager):
    """Quản lý bộ nhớ dài hạn bằng file text (TOCTOU-safe).

    Lưu và đọc nhận định từ file ``last_context.md``.

    Merge-on-write pattern:
        Khi save, hold lock → đọc file → thay thế section có cùng
        profile_name → ghi lại → release lock. Đảm bảo atomicity
        cho toàn bộ read-modify-write, tránh TOCTOU khi nhiều
        profile chạy song song.

    Args:
        memory_dir: Thư mục lưu file memory (default: ``./memory``).
        profile_name: Tên profile (vd: ``"oil"``, ``"crypto"``).
            Dùng để phân biệt sections trong cùng 1 file.
            Default: ``"default"``.
    """

    _FILENAME = "last_context.md"

    def __init__(
        self,
        memory_dir: Path | None = None,
        profile_name: str = "default",
    ) -> None:
        self._memory_dir = memory_dir or Path("./memory")
        self._file_path = self._memory_dir / self._FILENAME
        self._profile_name = profile_name
        self._lock = FileLock(
            str(self._file_path) + ".lock",
            timeout=_LOCK_TIMEOUT,
        )
        logger.info(
            "FileMemoryManager: %s (profile=%s)",
            self._file_path,
            self._profile_name,
        )

    def retrieve_last_context(self) -> str | None:
        """Đọc nhận định cũ từ file (thread/process-safe).

        Trả về section của profile hiện tại nếu có,
        hoặc toàn bộ nội dung nếu file không có sections.

        Returns:
            Nội dung nhận định nếu tồn tại, None nếu chưa có.

        Raises:
            RuntimeError: Khi không acquire được lock trong timeout.
        """
        if not self._file_path.exists():
            logger.info("  Chưa có file memory: %s", self._file_path)
            return None

        try:
            with self._lock:
                content = self._file_path.read_text(encoding="utf-8").strip()
        except Timeout:
            msg = f"Không thể acquire lock sau {_LOCK_TIMEOUT}s: {self._file_path}"
            logger.error(msg)
            raise RuntimeError(msg) from None

        if not content:
            return None

        # Thử tìm section riêng của profile này
        section = self._extract_section(content, self._profile_name)
        if section:
            logger.info(
                "  Đọc nhận định [%s]: %d chars từ %s",
                self._profile_name,
                len(section),
                self._file_path,
            )
            return section

        # Fallback: trả về toàn bộ (backward-compat với file cũ)
        logger.info(
            "  Đọc nhận định cũ (toàn bộ): %d chars từ %s",
            len(content),
            self._file_path,
        )
        return content

    def save_context(self, context_data: str) -> None:
        """Lưu nhận định mới ra file (TOCTOU-safe merge-on-write).

        Toàn bộ read-modify-write diễn ra bên trong lock:
        1. Đọc file hiện tại.
        2. Thay thế section cùng profile (hoặc thêm mới).
        3. Ghi lại file.

        Đảm bảo nhiều profile ghi song song không đè nhau.

        Args:
            context_data: Nội dung nhận định cần lưu.

        Raises:
            RuntimeError: Khi không thể ghi file hoặc lock timeout.
        """
        try:
            self._memory_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
            header = f"<!-- [{self._profile_name}] Saved: {timestamp} -->\n\n"
            new_section = header + context_data

            with self._lock:
                # ── Atomic read-modify-write ──
                existing = ""
                if self._file_path.exists():
                    existing = self._file_path.read_text(encoding="utf-8").strip()

                merged = self._merge_sections(existing, new_section)
                self._file_path.write_text(merged, encoding="utf-8")

            logger.info(
                "  Đã lưu nhận định [%s]: %d chars → %s",
                self._profile_name,
                len(context_data),
                self._file_path,
            )
        except Timeout:
            msg = f"Không thể acquire lock sau {_LOCK_TIMEOUT}s: {self._file_path}"
            logger.error(msg)
            raise RuntimeError(msg) from None
        except OSError as exc:
            msg = f"Không thể lưu memory file: {exc}"
            logger.error(msg)
            raise RuntimeError(msg) from None

    def retrieve_related_context(
        self,
        hot_keywords: list[str],
        max_results: int = 3,
    ) -> str | None:
        """File backend không hỗ trợ semantic retrieval theo keywords."""
        _ = hot_keywords, max_results
        return None

    # ── Private: Section management ──────────────────────────

    @staticmethod
    def _extract_section(content: str, profile_name: str) -> str | None:
        """Trích section của profile_name từ file content."""
        sections = content.split(_SECTION_SEPARATOR)
        for section in sections:
            match = _SECTION_HEADER_RE.search(section)
            if match and match.group(1) == profile_name:
                # Bỏ header, trả về nội dung
                lines = section.strip().split("\n", 2)
                return lines[-1].strip() if len(lines) > 1 else ""
        return None

    @staticmethod
    def _merge_sections(existing: str, new_section: str) -> str:
        """Merge new_section vào existing content.

        Nếu đã có section cùng profile → thay thế.
        Nếu chưa → append.
        """
        if not existing:
            return new_section

        # Tìm profile name trong new_section
        match = _SECTION_HEADER_RE.search(new_section)
        if not match:
            return new_section

        profile_name = match.group(1)

        # Split existing thành sections
        sections = existing.split(_SECTION_SEPARATOR)
        replaced = False
        result: list[str] = []

        for section in sections:
            section_match = _SECTION_HEADER_RE.search(section)
            if section_match and section_match.group(1) == profile_name:
                result.append(new_section)
                replaced = True
            else:
                result.append(section)

        if not replaced:
            result.append(new_section)

        return _SECTION_SEPARATOR.join(result)
