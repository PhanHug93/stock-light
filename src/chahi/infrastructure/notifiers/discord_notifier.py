"""Discord Notifier — gửi báo cáo ChaHi qua Discord Webhook.

Sử dụng HTTP POST tới Discord Webhook URL (không cần thư viện ngoài).
Tự động chia nhỏ message dài (Discord limit: 2000 chars).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import requests

from chahi.core.interfaces import INotifier

if TYPE_CHECKING:
    from pathlib import Path

    from chahi.core.entities import NotificationSettings

logger = logging.getLogger(__name__)

_MAX_MESSAGE_LENGTH: int = 2000  # Discord limit


class DiscordNotifier(INotifier):
    """Gửi báo cáo qua Discord Webhook.

    Args:
        settings: Cấu hình Discord (webhook_url).
    """

    def __init__(self, settings: NotificationSettings) -> None:
        self._webhook_url = settings.webhook_url
        logger.info("DiscordNotifier: webhook configured")

    def send_report(self, title: str, content: str) -> bool:
        """Gửi báo cáo tới Discord channel via webhook.

        Tự động chia nhỏ nếu nội dung > 2000 chars.
        Message đầu tiên có embed với tiêu đề, các phần sau là plain text.

        Args:
            title: Tiêu đề báo cáo.
            content: Nội dung Markdown.

        Returns:
            True nếu gửi thành công, False nếu có lỗi.
        """
        # Gửi embed header
        if not self._send_embed(title):
            return False

        # Gửi nội dung (chia nhỏ nếu cần)
        chunks = self._split_message(content)
        success = True
        for i, chunk in enumerate(chunks):
            if not self._send_text(chunk, i + 1, len(chunks)):
                success = False

        return success

    def _send_embed(self, title: str) -> bool:
        """Gửi embed header."""
        payload: dict[str, Any] = {
            "embeds": [
                {
                    "title": f"📊 {title}",
                    "color": 3447003,  # Blue
                    "footer": {"text": "ChaHi — Macro Report Generator"},
                }
            ],
        }
        return self._post(payload, "embed")

    def _send_text(self, text: str, part: int, total: int) -> bool:
        """Gửi một phần text."""
        payload: dict[str, Any] = {"content": text}
        return self._post(payload, f"text {part}/{total}")

    def send_file_attachment(self, file_path: Path, comment: str | None = None) -> bool:
        """Gửi file attachment qua Discord webhook.

        Args:
            file_path: Đường dẫn file cần gửi.
            comment: Nội dung text đi kèm file (optional).

        Returns:
            True nếu gửi thành công, False nếu thất bại.
        """
        if not file_path.exists() or not file_path.is_file():
            logger.warning("Discord attachment không tồn tại: %s", file_path)
            return False

        try:
            with file_path.open("rb") as f:
                response = requests.post(
                    self._webhook_url,
                    data={"content": comment or f"📎 {file_path.name}"},
                    files={"file": (file_path.name, f, "text/markdown")},
                    timeout=30,
                )

            if response.status_code in (200, 204):
                logger.info("  Discord: gửi file thành công (%s)", file_path.name)
                return True

            logger.warning(
                "Discord file upload error: status=%d, body=%s",
                response.status_code,
                response.text[:200],
            )
            return False
        except requests.RequestException as exc:
            logger.warning("Discord gửi file thất bại: %s", exc)
            return False

    def _post(self, payload: dict[str, Any], label: str) -> bool:
        """HTTP POST tới Discord webhook."""
        try:
            response = requests.post(
                self._webhook_url,
                json=payload,
                timeout=15,
            )

            if response.status_code in (200, 204):
                logger.info("  Discord: gửi thành công (%s)", label)
                return True

            logger.warning(
                "Discord Webhook error: status=%d, body=%s",
                response.status_code,
                response.text[:200],
            )
            return False

        except requests.RequestException as exc:
            logger.warning("Discord gửi thất bại: %s", exc)
            return False

    @staticmethod
    def _split_message(text: str) -> list[str]:
        """Chia message dài thành các phần ≤ 2000 chars.

        Ưu tiên cắt tại ``\\n\\n``, fallback ``\\n``.
        """
        if len(text) <= _MAX_MESSAGE_LENGTH:
            return [text]

        chunks: list[str] = []
        remaining = text

        while remaining:
            if len(remaining) <= _MAX_MESSAGE_LENGTH:
                chunks.append(remaining)
                break

            cut_at = _MAX_MESSAGE_LENGTH
            for sep in ("\n\n", "\n", ". "):
                idx = remaining.rfind(sep, 0, _MAX_MESSAGE_LENGTH)
                if idx > _MAX_MESSAGE_LENGTH // 2:
                    cut_at = idx + len(sep)
                    break

            chunks.append(remaining[:cut_at])
            remaining = remaining[cut_at:]

        return chunks
