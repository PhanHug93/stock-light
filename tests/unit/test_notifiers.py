"""Unit tests cho Notification subsystem (Phase 8).

Covers: NotificationManager (Composite), TelegramNotifier split,
DiscordNotifier split, NotificationSettings entity, và factory.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chahi.core.entities import NotificationSettings
from chahi.core.interfaces import INotifier
from chahi.infrastructure.notifiers.discord_notifier import DiscordNotifier
from chahi.infrastructure.notifiers.notification_manager import NotificationManager
from chahi.infrastructure.notifiers.notifier_factory import (
    create_notification_manager,
    create_notifiers,
)
from chahi.infrastructure.notifiers.telegram_notifier import TelegramNotifier

# ═════════════════════════════════════════════════════════════
# NotificationSettings Entity
# ═════════════════════════════════════════════════════════════


class TestNotificationSettings:
    """Test suite cho NotificationSettings entity."""

    def test_telegram_valid(self) -> None:
        """Cấu hình Telegram hợp lệ."""
        s = NotificationSettings(
            type="telegram",
            enabled=True,
            bot_token="123:ABC",
            chat_id="-100123",
        )
        assert s.type == "telegram"
        assert s.enabled is True
        assert s.bot_token == "123:ABC"

    def test_discord_valid(self) -> None:
        """Cấu hình Discord hợp lệ."""
        s = NotificationSettings(
            type="discord",
            enabled=True,
            webhook_url="https://discord.com/api/webhooks/123/abc",
        )
        assert s.type == "discord"
        assert s.webhook_url.startswith("https://")

    def test_telegram_missing_token_raises(self) -> None:
        """Telegram enabled nhưng thiếu bot_token phải raise ValueError."""
        with pytest.raises(ValueError, match="bot_token"):
            NotificationSettings(
                type="telegram",
                enabled=True,
                bot_token="",
                chat_id="-100123",
            )

    def test_telegram_missing_chat_id_raises(self) -> None:
        """Telegram enabled nhưng thiếu chat_id phải raise ValueError."""
        with pytest.raises(ValueError, match="chat_id"):
            NotificationSettings(
                type="telegram",
                enabled=True,
                bot_token="123:ABC",
                chat_id="",
            )

    def test_discord_missing_webhook_raises(self) -> None:
        """Discord enabled nhưng thiếu webhook_url phải raise ValueError."""
        with pytest.raises(ValueError, match="webhook_url"):
            NotificationSettings(
                type="discord",
                enabled=True,
                webhook_url="",
            )

    def test_invalid_type_raises(self) -> None:
        """Type không hợp lệ phải raise ValueError."""
        with pytest.raises(ValueError, match="type"):
            NotificationSettings(type="slack", enabled=False)

    def test_disabled_skips_validation(self) -> None:
        """Khi enabled=False, không validate token/webhook."""
        s = NotificationSettings(
            type="telegram",
            enabled=False,
            bot_token="",
            chat_id="",
        )
        assert s.enabled is False

    def test_frozen(self) -> None:
        """Không thể thay đổi attribute."""
        s = NotificationSettings(type="telegram", enabled=False)
        with pytest.raises(AttributeError):
            s.enabled = True  # type: ignore[misc]


# ═════════════════════════════════════════════════════════════
# NotificationManager (Composite Pattern)
# ═════════════════════════════════════════════════════════════


class TestNotificationManager:
    """Test suite cho NotificationManager Composite."""

    def test_empty_manager_returns_true(self) -> None:
        """Manager rỗng (không có client) trả về True."""
        manager = NotificationManager(clients=[])
        assert manager.send_report("Test", "Content") is True

    def test_client_count(self) -> None:
        """client_count trả về số lượng client đăng ký."""
        mock1 = MagicMock(spec=INotifier)
        mock2 = MagicMock(spec=INotifier)
        manager = NotificationManager(clients=[mock1, mock2])
        assert manager.client_count == 2

    def test_all_success(self) -> None:
        """Tất cả client thành công → True."""
        mock1 = MagicMock(spec=INotifier)
        mock1.send_report.return_value = True
        mock2 = MagicMock(spec=INotifier)
        mock2.send_report.return_value = True

        manager = NotificationManager(clients=[mock1, mock2])
        result = manager.send_report("Title", "Content")

        assert result is True
        mock1.send_report.assert_called_once_with("Title", "Content")
        mock2.send_report.assert_called_once_with("Title", "Content")

    def test_one_failure_still_sends_all(self) -> None:
        """Một client fail → vẫn gửi client khác, return False."""
        mock_ok = MagicMock(spec=INotifier)
        mock_ok.send_report.return_value = True
        mock_fail = MagicMock(spec=INotifier)
        mock_fail.send_report.return_value = False

        manager = NotificationManager(clients=[mock_fail, mock_ok])
        result = manager.send_report("Title", "Content")

        assert result is False
        # QUAN TRỌNG: cả 2 đều được gọi, không dừng ở client đầu
        mock_ok.send_report.assert_called_once()
        mock_fail.send_report.assert_called_once()

    def test_exception_isolated(self) -> None:
        """Client raise exception → không ảnh hưởng client khác."""
        mock_crash = MagicMock(spec=INotifier)
        mock_crash.send_report.side_effect = ConnectionError("Network down")
        mock_ok = MagicMock(spec=INotifier)
        mock_ok.send_report.return_value = True

        manager = NotificationManager(clients=[mock_crash, mock_ok])
        result = manager.send_report("Title", "Content")

        assert result is False  # overall fail do crash
        mock_ok.send_report.assert_called_once()  # vẫn send được

    def test_all_crash_returns_false(self) -> None:
        """Tất cả crash → return False."""
        mock1 = MagicMock(spec=INotifier)
        mock1.send_report.side_effect = RuntimeError("Boom")
        mock2 = MagicMock(spec=INotifier)
        mock2.send_report.side_effect = TimeoutError("Timeout")

        manager = NotificationManager(clients=[mock1, mock2])
        result = manager.send_report("Title", "Content")

        assert result is False


# ═════════════════════════════════════════════════════════════
# TelegramNotifier — Message Splitting
# ═════════════════════════════════════════════════════════════


class TestTelegramSplit:
    """Test _split_message() của TelegramNotifier."""

    def test_short_message_no_split(self) -> None:
        """Message ngắn hơn 4096 chars → 1 chunk."""
        text = "Hello world"
        chunks = TelegramNotifier._split_message(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_exact_limit_no_split(self) -> None:
        """Message đúng 4096 chars → 1 chunk."""
        text = "A" * 4096
        chunks = TelegramNotifier._split_message(text)
        assert len(chunks) == 1

    def test_long_message_splits(self) -> None:
        """Message > 4096 chars → nhiều chunks."""
        # Tạo text với paragraph breaks
        paragraphs = ["Paragraph " + str(i) + "." * 200 for i in range(30)]
        text = "\n\n".join(paragraphs)
        assert len(text) > 4096

        chunks = TelegramNotifier._split_message(text)
        assert len(chunks) > 1
        # Tất cả chunks <= 4096
        for chunk in chunks:
            assert len(chunk) <= 4096
        # Ghép lại phải bằng text gốc
        assert "".join(chunks) == text

    def test_split_prefers_paragraph_break(self) -> None:
        """Ưu tiên cắt tại \\n\\n."""
        block = "A" * 2000 + "\n\n" + "B" * 2000 + "\n\n" + "C" * 2000
        chunks = TelegramNotifier._split_message(block)
        assert len(chunks) >= 2
        # Chunk đầu phải kết thúc tại paragraph break
        assert chunks[0].endswith("\n\n")


# ═════════════════════════════════════════════════════════════
# DiscordNotifier — Message Splitting
# ═════════════════════════════════════════════════════════════


class TestDiscordSplit:
    """Test _split_message() của DiscordNotifier."""

    def test_short_message_no_split(self) -> None:
        """Message ngắn hơn 2000 chars → 1 chunk."""
        text = "Hello world"
        chunks = DiscordNotifier._split_message(text)
        assert len(chunks) == 1

    def test_long_message_splits(self) -> None:
        """Message > 2000 chars → nhiều chunks."""
        paragraphs = ["Line " + str(i) + "." * 100 for i in range(30)]
        text = "\n\n".join(paragraphs)
        assert len(text) > 2000

        chunks = DiscordNotifier._split_message(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 2000
        assert "".join(chunks) == text


# ═════════════════════════════════════════════════════════════
# Notifier Factory
# ═════════════════════════════════════════════════════════════


class TestNotifierFactory:
    """Test create_notifiers() và create_notification_manager()."""

    def test_empty_config_returns_empty(self) -> None:
        """Config rỗng → list rỗng."""
        result = create_notifiers({})
        assert result == []

    def test_none_config_returns_empty(self) -> None:
        """Config None-ish → list rỗng."""
        result = create_notifiers({})
        assert result == []

    def test_telegram_enabled(self) -> None:
        """Telegram enabled → 1 TelegramNotifier."""
        config = {
            "telegram": {
                "enabled": True,
                "bot_token": "123:TOKEN",
                "chat_id": "-100123",
            },
        }
        result = create_notifiers(config)
        assert len(result) == 1
        assert isinstance(result[0], TelegramNotifier)

    def test_discord_enabled(self) -> None:
        """Discord enabled → 1 DiscordNotifier."""
        config = {
            "discord": {
                "enabled": True,
                "webhook_url": "https://discord.com/api/webhooks/123/abc",
            },
        }
        result = create_notifiers(config)
        assert len(result) == 1
        assert isinstance(result[0], DiscordNotifier)

    def test_both_enabled(self) -> None:
        """Cả hai enabled → 2 notifiers."""
        config = {
            "telegram": {
                "enabled": True,
                "bot_token": "123:TOKEN",
                "chat_id": "-100123",
            },
            "discord": {
                "enabled": True,
                "webhook_url": "https://discord.com/api/webhooks/123/abc",
            },
        }
        result = create_notifiers(config)
        assert len(result) == 2

    def test_all_disabled(self) -> None:
        """Cả hai disabled → list rỗng."""
        config = {
            "telegram": {"enabled": False},
            "discord": {"enabled": False},
        }
        result = create_notifiers(config)
        assert result == []

    def test_invalid_telegram_config_skipped(self) -> None:
        """Cấu hình Telegram lỗi → bỏ qua, không crash."""
        config = {
            "telegram": {
                "enabled": True,
                "bot_token": "",  # Missing → ValueError
                "chat_id": "-100123",
            },
        }
        result = create_notifiers(config)
        assert result == []  # Bỏ qua, không crash

    def test_create_notification_manager_returns_manager(self) -> None:
        """create_notification_manager trả về NotificationManager."""
        config = {
            "telegram": {
                "enabled": True,
                "bot_token": "123:TOKEN",
                "chat_id": "-100123",
            },
        }
        manager = create_notification_manager(config)
        assert isinstance(manager, NotificationManager)
        assert manager.client_count == 1

    def test_create_notification_manager_empty(self) -> None:
        """Empty config → NotificationManager rỗng."""
        manager = create_notification_manager({})
        assert isinstance(manager, NotificationManager)
        assert manager.client_count == 0


# ═════════════════════════════════════════════════════════════
# TelegramNotifier — Markdown Fallback
# ═════════════════════════════════════════════════════════════


class TestTelegramFallback:
    """Test Telegram Markdown parse_mode fallback."""

    @pytest.fixture()
    def notifier(self) -> TelegramNotifier:
        """Tạo TelegramNotifier mẫu."""
        settings = NotificationSettings(
            type="telegram",
            enabled=True,
            bot_token="123:FAKE",
            chat_id="-100123",
        )
        return TelegramNotifier(settings)

    @patch("chahi.infrastructure.notifiers.telegram_notifier.requests.post")
    def test_markdown_success(
        self, mock_post: MagicMock, notifier: TelegramNotifier
    ) -> None:
        """Markdown parse thành công → không retry."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = notifier.send_report("Title", "Content")
        assert result is True
        assert mock_post.call_count == 1  # Chỉ 1 lần

    @patch("chahi.infrastructure.notifiers.telegram_notifier.requests.post")
    def test_markdown_400_fallback_success(
        self, mock_post: MagicMock, notifier: TelegramNotifier
    ) -> None:
        """Markdown parse lỗi 400 → retry plain text thành công."""
        mock_fail = MagicMock()
        mock_fail.status_code = 400
        mock_fail.text = "Bad Request: can't parse entities"

        mock_ok = MagicMock()
        mock_ok.status_code = 200

        mock_post.side_effect = [mock_fail, mock_ok]

        result = notifier.send_report("Title", "Short content")
        assert result is True
        assert mock_post.call_count == 2  # Retry 1 lần

    @patch("chahi.infrastructure.notifiers.telegram_notifier.requests.post")
    def test_network_error_returns_false(
        self, mock_post: MagicMock, notifier: TelegramNotifier
    ) -> None:
        """Network error → return False, no crash."""
        import requests as req

        mock_post.side_effect = req.ConnectionError("Network down")

        result = notifier.send_report("Title", "Content")
        assert result is False
