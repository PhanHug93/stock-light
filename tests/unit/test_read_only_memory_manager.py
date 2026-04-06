"""Unit tests cho ReadOnlyMemoryManager."""

from __future__ import annotations

from unittest.mock import MagicMock

from chahi.infrastructure.memory.read_only_memory_manager import ReadOnlyMemoryManager


class TestReadOnlyMemoryManager:
    """Tests cho chế độ memory read-only (không save)."""

    def test_retrieve_delegates_to_underlying_manager(self) -> None:
        delegate = MagicMock()
        delegate.retrieve_last_context.return_value = "old-context"

        mgr = ReadOnlyMemoryManager(delegate=delegate)
        result = mgr.retrieve_last_context()

        assert result == "old-context"
        delegate.retrieve_last_context.assert_called_once_with()

    def test_save_context_is_noop(self) -> None:
        delegate = MagicMock()

        mgr = ReadOnlyMemoryManager(delegate=delegate)
        mgr.save_context("new-context")

        delegate.save_context.assert_not_called()

    def test_close_delegates(self) -> None:
        delegate = MagicMock()

        mgr = ReadOnlyMemoryManager(delegate=delegate)
        mgr.close()

        delegate.close.assert_called_once_with()
