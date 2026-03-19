"""Token Counter — provider-aware token estimation for LLM budget.

Hỗ trợ nhiều LLM providers với tokenizer riêng:
    - ``gemini``: dùng cl100k_base (conservative, ~tương đương)
    - ``lm_studio`` (Qwen/local): dùng heuristic Vietnamese-optimized
      vì Qwen có vocab ~151K tokens, tối ưu tiếng Việt hơn GPT-4.

**Tại sao không dùng cl100k_base cho tất cả?**
    cl100k_base là vocabulary của GPT-4 (~100K tokens). Qwen 2.5
    có vocab lớn hơn (~151K) và encode tiếng Việt hiệu quả hơn.
    Dùng cl100k_base cho Qwen sẽ over-estimate ~40%, lãng phí
    context window.

Truncation:
    Khi có tiktoken: ``encoder.decode(encoder.encode(text)[:max_tokens])``
    — an toàn Unicode, O(1) lần encode.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Module-level state ──
_encoder = None
_TIKTOKEN_AVAILABLE = True
_current_provider: str = "gemini"  # default

# Heuristic ratios: chars per token (conservative = ít chars/token = over-estimate)
_HEURISTIC_RATIOS: dict[str, float] = {
    "gemini": 1.7,  # cl100k_base ~1 token / 1.7 chars cho VN
    "lm_studio": 2.5,  # Qwen vocab lớn, encode VN hiệu quả hơn
}


def configure(provider: str = "gemini") -> None:
    """Cấu hình token counter cho LLM provider.

    Gọi hàm này trước khi dùng estimate_tokens/truncate.
    Reset encoder cache khi đổi provider.

    Args:
        provider: ``"gemini"`` hoặc ``"lm_studio"``.
    """
    global _current_provider, _encoder  # noqa: PLW0603
    if provider != _current_provider:
        _encoder = None  # Reset cache khi đổi provider
    _current_provider = provider
    logger.debug("Token counter configured: provider=%s", provider)


def _get_encoder():  # noqa: ANN202
    """Lazy-load tiktoken encoder (chỉ cho Gemini/OpenAI-compatible)."""
    global _encoder, _TIKTOKEN_AVAILABLE  # noqa: PLW0603

    if _encoder is not None:
        return _encoder

    # Chỉ dùng tiktoken cho Gemini (cl100k_base ~ equivalent)
    # Qwen/local LLM: dùng heuristic vì tiktoken không có vocab Qwen
    if _current_provider != "gemini":
        return None

    try:
        import tiktoken  # type: ignore[import-untyped]

        _encoder = tiktoken.get_encoding("cl100k_base")
        return _encoder
    except (ImportError, Exception):
        _TIKTOKEN_AVAILABLE = False
        logger.debug("tiktoken not available, using heuristic.")
        return None


def _get_ratio() -> float:
    """Lấy heuristic ratio cho provider hiện tại."""
    return _HEURISTIC_RATIOS.get(_current_provider, 1.7)


def estimate_tokens(text: str) -> int:
    """Ước lượng số tokens cho một chuỗi text.

    - Gemini: dùng tiktoken cl100k_base (chính xác).
    - Qwen/local: dùng heuristic ratio (conservative).

    Args:
        text: Chuỗi text cần ước lượng.

    Returns:
        Số tokens ước lượng.
    """
    if not text:
        return 0

    encoder = _get_encoder()
    if encoder is not None:
        return len(encoder.encode(text))

    # Heuristic: provider-specific ratio
    ratio = _get_ratio()
    return int(len(text) / ratio) + 1


def truncate_to_token_budget(
    text: str,
    max_tokens: int,
) -> str:
    """Cắt text sao cho không vượt quá token budget.

    - Gemini (tiktoken): encode → slice → decode (Unicode-safe).
    - Qwen/local: heuristic chars + word boundary.

    Args:
        text: Chuỗi text cần cắt.
        max_tokens: Ngân sách token tối đa.

    Returns:
        Text đã cắt + "…" nếu vượt budget, giữ nguyên nếu không.
    """
    if not text:
        return text

    encoder = _get_encoder()

    if encoder is not None:
        tokens = encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        # Cắt token array → decode: Unicode-safe, O(1) encode
        truncated = encoder.decode(tokens[:max_tokens])
        # Cắt tại word boundary nếu có thể
        cut = truncated.rfind(" ")
        if cut > len(truncated) // 2:
            truncated = truncated[:cut]
        return truncated.rstrip() + "…"

    # ── Fallback: heuristic ──
    ratio = _get_ratio()
    estimated = int(len(text) / ratio) + 1
    if estimated <= max_tokens:
        return text

    # Ước lượng char budget từ token budget
    char_budget = int(max_tokens * ratio)
    cut = text.rfind(" ", 0, char_budget)
    if cut <= 0:
        cut = char_budget
    return text[:cut].rstrip() + "…"
