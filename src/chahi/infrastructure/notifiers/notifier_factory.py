"""Notifier Factory — tạo INotifier từ config.

Đọc section ``notifications`` trong config YAML và khởi tạo
các notifier tương ứng (Telegram, Discord) nếu ``enabled: true``.

Cung cấp hai hàm:
- ``create_notifiers()``: Trả về ``list[INotifier]`` (backward compatible).
- ``create_notification_manager()``: Trả về ``NotificationManager`` (Composite).
"""

from __future__ import annotations

import logging
from typing import Any

from chahi.core.entities import NotificationSettings
from chahi.core.interfaces import INotifier
from chahi.infrastructure.notifiers.discord_notifier import DiscordNotifier
from chahi.infrastructure.notifiers.notification_manager import NotificationManager
from chahi.infrastructure.notifiers.telegram_notifier import TelegramNotifier

logger = logging.getLogger(__name__)


def create_notifiers(raw_config: dict[str, Any]) -> list[INotifier]:
    """Tạo danh sách notifier từ raw config dict.

    Args:
        raw_config: Dict từ ``yaml.safe_load(config_file)["notifications"]``.
            Ví dụ::

                {
                    "telegram": {"enabled": True, "bot_token": "...", "chat_id": "..."},
                    "discord":  {"enabled": True, "webhook_url": "..."},
                }

    Returns:
        Danh sách INotifier đã khởi tạo (chỉ các kênh enabled).
        Trả về list rỗng nếu không có kênh nào enabled.
    """
    if not raw_config:
        return []

    notifiers: list[INotifier] = []

    # ── Telegram ──
    tg_config = raw_config.get("telegram", {})
    if isinstance(tg_config, dict) and tg_config.get("enabled", False):
        try:
            settings = NotificationSettings(
                type="telegram",
                enabled=True,
                bot_token=str(tg_config.get("bot_token", "")),
                chat_id=str(tg_config.get("chat_id", "")),
            )
            notifiers.append(TelegramNotifier(settings))
            logger.info("✓ Telegram notifier enabled")
        except ValueError as exc:
            logger.warning("Telegram config lỗi: %s", exc)

    # ── Discord ──
    dc_config = raw_config.get("discord", {})
    if isinstance(dc_config, dict) and dc_config.get("enabled", False):
        try:
            settings = NotificationSettings(
                type="discord",
                enabled=True,
                webhook_url=str(dc_config.get("webhook_url", "")),
            )
            notifiers.append(DiscordNotifier(settings))
            logger.info("✓ Discord notifier enabled")
        except ValueError as exc:
            logger.warning("Discord config lỗi: %s", exc)

    return notifiers


def create_notification_manager(
    raw_config: dict[str, Any],
) -> NotificationManager:
    """Tạo NotificationManager (Composite) từ raw config dict.

    Khởi tạo các notifier enabled rồi gói vào NotificationManager.
    NotificationManager đảm bảo fail-safe giữa các nền tảng.

    Args:
        raw_config: Dict từ ``yaml.safe_load(config_file)["notifications"]``.

    Returns:
        NotificationManager chứa tất cả client đã enabled.
        Nếu không có kênh nào enabled, trả về manager rỗng (không lỗi).
    """
    clients = create_notifiers(raw_config)
    return NotificationManager(clients=clients)
