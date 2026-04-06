"""Recording LLM client — dump input trước khi gọi provider API."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from threading import Lock
from typing import TYPE_CHECKING

from chahi.core.interfaces import ILLMClient

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class RecordingLLMClient(ILLMClient):
    """Decorator cho ILLMClient: ghi lại input mỗi lần analyze().

    Dùng cho debug/fallback: có thể copy input này để request ở provider khác.
    """

    def __init__(
        self,
        delegate: ILLMClient,
        dump_dir: Path,
    ) -> None:
        self._delegate = delegate
        self._dump_dir = dump_dir
        self._counter = 0
        self._lock = Lock()
        self._dump_dir.mkdir(parents=True, exist_ok=True)

    @property
    def supports_concurrency(self) -> bool:
        return self._delegate.supports_concurrency

    def analyze(self, system_prompt: str, user_content: str) -> str:
        self._dump_input(system_prompt, user_content)
        return self._delegate.analyze(
            system_prompt=system_prompt,
            user_content=user_content,
        )

    def _dump_input(self, system_prompt: str, user_content: str) -> None:
        with self._lock:
            self._counter += 1
            call_no = self._counter

        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        file_path = self._dump_dir / f"llm_input_{call_no:03d}_{ts}.md"

        payload = (
            f"# LLM Input #{call_no}\n\n"
            f"- Generated at (UTC): {datetime.now(UTC).isoformat()}\n\n"
            "## System Prompt\n\n"
            "```text\n"
            f"{system_prompt}\n"
            "```\n\n"
            "## User Content\n\n"
            "```text\n"
            f"{user_content}\n"
            "```\n"
        )

        try:
            file_path.write_text(payload, encoding="utf-8")
            logger.info("Đã dump LLM input: %s", file_path.resolve())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Không thể dump LLM input (%s): %s", file_path, exc)
