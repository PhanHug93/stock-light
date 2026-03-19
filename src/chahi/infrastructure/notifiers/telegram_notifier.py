"""Telegram Notifier — gửi báo cáo ChaHi qua Telegram Bot API.

Sử dụng HTTP POST tới ``api.telegram.org`` (không cần thư viện ngoài).
Tự động chia nhỏ message dài (Telegram limit: 4096 chars).
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from chahi.core.entities import NotificationSettings
from chahi.core.interfaces import INotifier

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_MAX_MESSAGE_LENGTH: int = 4096  # Telegram limit


class TelegramNotifier(INotifier):
    """Gửi báo cáo qua Telegram Bot API.

    Args:
        settings: Cấu hình Telegram (bot_token, chat_id).
    """

    def __init__(self, settings: NotificationSettings) -> None:
        self._token = settings.bot_token
        self._chat_id = settings.chat_id
        self._url = _TELEGRAM_API.format(token=self._token)
        logger.info("TelegramNotifier: chat_id=%s", self._chat_id)

    def send_report(self, title: str, content: str) -> bool:
        """Gửi báo cáo Markdown tới Telegram chat.

        Tự động chia nhỏ nếu nội dung > 4096 chars.

        Args:
            title: Tiêu đề báo cáo.
            content: Nội dung Markdown.

        Returns:
            True nếu gửi thành công (tất cả phần), False nếu có lỗi.
        """
        full_text = f"📊 *{title}*\n\n{content}"
        chunks = self._split_message(full_text)

        success = True
        for i, chunk in enumerate(chunks):
            if not self._send_chunk(chunk, i + 1, len(chunks)):
                success = False

        return success

    def _send_chunk(self, text: str, part: int, total: int) -> bool:
        """Gửi một phần message tới Telegram.

        Thử gửi với parse_mode=Markdown trước. Nếu Telegram trả 400
        (lỗi parse Markdown strict), tự động retry không parse_mode.
        """
        payload: dict[str, Any] = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }

        try:
            response = requests.post(self._url, json=payload, timeout=15)

            if response.status_code == 200:
                logger.info(
                    "  Telegram: gửi thành công (%d/%d)",
                    part,
                    total,
                )
                return True

            # Fallback: Markdown parse lỗi → gửi lại không parse_mode
            if response.status_code == 400:
                logger.warning(
                    "Telegram Markdown parse lỗi, retry không parse_mode...",
                )
                payload.pop("parse_mode")
                retry = requests.post(self._url, json=payload, timeout=15)
                if retry.status_code == 200:
                    logger.info(
                        "  Telegram: gửi thành công (plain text, %d/%d)",
                        part,
                        total,
                    )
                    return True

            logger.warning(
                "Telegram API error: status=%d, body=%s",
                response.status_code,
                response.text[:200],
            )
            return False

        except requests.RequestException as exc:
            logger.warning("Telegram gửi thất bại: %s", exc)
            return False

    @staticmethod
    def _split_message(text: str) -> list[str]:
        """Chia message dài thành các phần ≤ 4096 chars.

        Ưu tiên cắt tại ``\\n\\n`` (paragraph break), fallback ``\\n``.
        """
        if len(text) <= _MAX_MESSAGE_LENGTH:
            return [text]

        chunks: list[str] = []
        remaining = text

        while remaining:
            if len(remaining) <= _MAX_MESSAGE_LENGTH:
                chunks.append(remaining)
                break

            # Tìm điểm cắt tốt nhất
            cut_at = _MAX_MESSAGE_LENGTH
            for sep in ("\n\n", "\n", ". "):
                idx = remaining.rfind(sep, 0, _MAX_MESSAGE_LENGTH)
                if idx > _MAX_MESSAGE_LENGTH // 2:
                    cut_at = idx + len(sep)
                    break

            chunks.append(remaining[:cut_at])
            remaining = remaining[cut_at:]

        return chunks
