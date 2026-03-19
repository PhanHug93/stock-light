"""Memory Factory — dispatch IMemoryManager theo config.

Tuân thủ Open/Closed Principle: thêm backend mới chỉ cần
thêm entry vào factory, không sửa code cũ.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from chahi.infrastructure.memory.file_memory_manager import FileMemoryManager
from chahi.infrastructure.memory.mcp_memory_manager import MCPHttpMemoryManager

if TYPE_CHECKING:
    from chahi.core.entities import MemorySettings
    from chahi.core.interfaces import IMemoryManager

logger = logging.getLogger(__name__)


def create_memory_manager(settings: MemorySettings) -> IMemoryManager:
    """Factory: tạo Memory Manager theo type trong settings.

    Args:
        settings: Cấu hình Memory đã validate.

    Returns:
        Instance của IMemoryManager phù hợp.

    Raises:
        ValueError: Khi type không được hỗ trợ.
    """
    if settings.type == "file":
        logger.info("Memory backend: file (./memory/)")
        return FileMemoryManager(memory_dir=Path("./memory"))

    if settings.type == "mcp":
        logger.info("Memory backend: MCP HTTP (%s)", settings.url)
        return MCPHttpMemoryManager(settings=settings)

    raise ValueError(
        f"Memory type không hỗ trợ: '{settings.type}'. Hỗ trợ: ['file', 'mcp']"
    )
