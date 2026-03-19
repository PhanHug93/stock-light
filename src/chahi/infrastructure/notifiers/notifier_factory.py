"""Notifier Factory — tạo INotifier từ NotificationSettings.

Nhận domain entities (NotificationSettings) từ IConfigReader và
khởi tạo các notifier tương ứng (Telegram, Discord) nếu ``enabled``.

Cung cấp hai hàm:
- ``create_notifiers()``: Trả về ``list[INotifier]``.
- ``create_notification_manager()``: Trả về ``NotificationManager`` (Composite).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from chahi.infrastructure.notifiers.discord_notifier import DiscordNotifier
from chahi.infrastructure.notifiers.notification_manager import NotificationManager
from chahi.infrastructure.notifiers.telegram_notifier import TelegramNotifier

if TYPE_CHECKING:
    from chahi.core.entities import NotificationSettings
    from chahi.core.interfaces import INotifier

logger = logging.getLogger(__name__)


def create_notifiers(
    settings: list[NotificationSettings],
) -> list[INotifier]:
    """Tạo danh sách notifier từ NotificationSettings entities.

    Args:
        settings: Danh sách NotificationSettings từ ``IConfigReader``.

    Returns:
        Danh sách INotifier đã khởi tạo (chỉ các kênh enabled).
        Trả về list rỗng nếu không có kênh nào enabled.
    """
    notifiers: list[INotifier] = []

    for setting in settings:
        if not setting.enabled:
            continue

        try:
            if setting.type == "telegram":
                notifiers.append(TelegramNotifier(setting))
                logger.info("✓ Telegram notifier enabled")
            elif setting.type == "discord":
                notifiers.append(DiscordNotifier(setting))
                logger.info("✓ Discord notifier enabled")
            else:
                logger.warning(
                    "Notifier type không hỗ trợ: %s", setting.type
                )
        except ValueError as exc:
            logger.warning("%s config lỗi: %s", setting.type, exc)

    return notifiers


def create_notification_manager(
    settings: list[NotificationSettings],
) -> NotificationManager:
    """Tạo NotificationManager (Composite) từ NotificationSettings.

    Args:
        settings: Danh sách NotificationSettings từ ``IConfigReader``.

    Returns:
        NotificationManager chứa tất cả client đã enabled.
        Nếu không có kênh nào enabled, trả về manager rỗng.
    """
    clients = create_notifiers(settings)
    return NotificationManager(clients=clients)
