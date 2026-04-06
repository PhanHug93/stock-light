"""Read-only Memory Manager — chỉ đọc, không lưu context mới."""

from __future__ import annotations

import logging

from chahi.core.interfaces import IMemoryManager

logger = logging.getLogger(__name__)


class ReadOnlyMemoryManager(IMemoryManager):
    """Decorator cho IMemoryManager: disable save để tránh bẩn memory."""

    def __init__(self, delegate: IMemoryManager) -> None:
        self._delegate = delegate

    def retrieve_last_context(self) -> str | None:
        return self._delegate.retrieve_last_context()

    def save_context(self, context_data: str) -> None:
        logger.info(
            (
                "Memory store disabled (--no-memory-store): "
                "bỏ qua save_context (%d chars)."
            ),
            len(context_data),
        )

    def retrieve_related_context(
        self,
        hot_keywords: list[str],
        max_results: int = 3,
    ) -> str | None:
        return self._delegate.retrieve_related_context(
            hot_keywords=hot_keywords,
            max_results=max_results,
        )

    def close(self) -> None:
        self._delegate.close()
