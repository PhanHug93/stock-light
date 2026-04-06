"""Capture-only LLM client — dump input mà không gọi provider API."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from threading import Lock
from typing import TYPE_CHECKING

from chahi.core.interfaces import ILLMClient

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class CaptureOnlyLLMClient(ILLMClient):
    """Client giả lập: chỉ lưu input để dùng ở nơi khác, không gọi API thật."""

    def __init__(self, dump_dir: Path) -> None:
        self._dump_dir = dump_dir
        self._counter = 0
        self._lock = Lock()
        self._dump_dir.mkdir(parents=True, exist_ok=True)

    @property
    def supports_concurrency(self) -> bool:
        """Giữ chế độ tuần tự để thứ tự dump dễ theo dõi."""
        return False

    def analyze(self, system_prompt: str, user_content: str) -> str:
        file_path = self._dump_input(system_prompt, user_content)
        return f"[CAPTURE_ONLY] Đã lưu input, không gọi provider API. File: {file_path}"

    def _dump_input(self, system_prompt: str, user_content: str) -> Path:
        with self._lock:
            self._counter += 1
            call_no = self._counter

        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        file_path = self._dump_dir / f"llm_input_{call_no:03d}_{ts}.md"

        payload = (
            f"# LLM Input #{call_no}\n\n"
            f"- Generated at (UTC): {datetime.now(UTC).isoformat()}\n"
            "- Mode: CAPTURE_ONLY (no provider call)\n\n"
            "## System Prompt\n\n"
            "```text\n"
            f"{system_prompt}\n"
            "```\n\n"
            "## User Content\n\n"
            "```text\n"
            f"{user_content}\n"
            "```\n"
        )

        file_path.write_text(payload, encoding="utf-8")
        resolved = file_path.resolve()
        logger.info("Đã dump LLM input (capture-only): %s", resolved)
        return resolved
