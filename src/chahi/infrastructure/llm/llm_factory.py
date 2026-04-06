"""LLM Client Factory — Strategy Pattern dispatch.

Tạo LLM client phù hợp dựa trên ``provider`` trong LLMSettings.
Tuân thủ Open/Closed Principle: thêm provider mới chỉ cần
thêm entry vào ``_PROVIDERS`` dict, không sửa code cũ.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from chahi.infrastructure.llm.gemini_client import GeminiClient
from chahi.infrastructure.llm.lm_studio_client import LMStudioClient
from chahi.infrastructure.llm.openai_client import OpenAIClient

if TYPE_CHECKING:
    from collections.abc import Callable

    from chahi.core.entities import LLMSettings
    from chahi.core.interfaces import ILLMClient

logger = logging.getLogger(__name__)

# ── Provider registry ──────────────────────────────────────
# Mỗi factory function nhận LLMSettings và trả về ILLMClient.

_PROVIDERS: dict[str, Callable[[LLMSettings], ILLMClient]] = {
    "lm_studio": lambda s: LMStudioClient(settings=s),
    "gemini": lambda s: GeminiClient(settings=s),
    "openai": lambda s: OpenAIClient(settings=s),
}


def create_llm_client(settings: LLMSettings) -> ILLMClient:
    """Factory: tạo LLM client theo provider trong settings.

    Args:
        settings: Cấu hình LLM đã validate (chứa provider field).

    Returns:
        Instance của ILLMClient phù hợp với provider.

    Raises:
        ValueError: Khi provider không được hỗ trợ.
    """
    factory_fn = _PROVIDERS.get(settings.provider)

    if factory_fn is None:
        raise ValueError(
            f"LLM provider không hỗ trợ: '{settings.provider}'. "
            f"Hỗ trợ: {list(_PROVIDERS.keys())}"
        )

    logger.info("Khởi tạo LLM client: provider=%s", settings.provider)
    return factory_fn(settings)
