"""File Memory Manager — fallback implementation of IMemoryManager.

Lưu nhận định phân tích ra file ``last_context.md`` trên đĩa.
Hoạt động ngay lập tức, không cần server hay dependency bên ngoài.

Dùng làm fallback khi chưa cắm MCP server thật.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from chahi.core.interfaces import IMemoryManager

logger = logging.getLogger(__name__)


class FileMemoryManager(IMemoryManager):
    """Quản lý bộ nhớ dài hạn bằng file text.

    Lưu và đọc nhận định từ file ``last_context.md``.
    Mỗi lần save sẽ ghi đè file cũ (chỉ giữ nhận định gần nhất).

    Args:
        memory_dir: Thư mục lưu file memory (default: ``./memory``).
    """

    _FILENAME = "last_context.md"

    def __init__(self, memory_dir: Path | None = None) -> None:
        self._memory_dir = memory_dir or Path("./memory")
        self._file_path = self._memory_dir / self._FILENAME
        logger.info("FileMemoryManager: %s", self._file_path)

    def retrieve_last_context(self) -> str | None:
        """Đọc nhận định cũ từ file.

        Returns:
            Nội dung file nếu tồn tại, None nếu chưa có.
        """
        if not self._file_path.exists():
            logger.info("  Chưa có file memory: %s", self._file_path)
            return None

        content = self._file_path.read_text(encoding="utf-8").strip()
        if not content:
            return None

        logger.info(
            "  Đọc nhận định cũ: %d chars từ %s",
            len(content),
            self._file_path,
        )
        return content

    def save_context(self, context_data: str) -> None:
        """Lưu nhận định mới ra file.

        Tạo thư mục nếu chưa tồn tại. Thêm timestamp vào header.

        Args:
            context_data: Nội dung nhận định cần lưu.

        Raises:
            RuntimeError: Khi không thể ghi file.
        """
        try:
            self._memory_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
            header = f"<!-- Saved: {timestamp} -->\n\n"
            full_content = header + context_data

            self._file_path.write_text(full_content, encoding="utf-8")
            logger.info(
                "  Đã lưu nhận định: %d chars → %s",
                len(context_data),
                self._file_path,
            )
        except OSError as exc:
            msg = f"Không thể lưu memory file: {exc}"
            logger.error(msg)
            raise RuntimeError(msg) from None
