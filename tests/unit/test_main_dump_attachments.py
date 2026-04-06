"""Unit tests cho helper gửi Discord dump attachments trong main.py."""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

from main import _collect_dump_files_since, _send_discord_dump_attachments

from chahi.core.entities import NotificationSettings


class TestCollectDumpFilesSince:
    """Tests cho lọc dump files theo thời gian."""

    def test_returns_only_new_files(self, tmp_path) -> None:
        old_file = tmp_path / "llm_input_001_old.md"
        old_file.write_text("old", encoding="utf-8")
        old_time = time.time() - 60
        os.utime(old_file, (old_time, old_time))

        since_epoch = time.time()

        new_file = tmp_path / "llm_input_002_new.md"
        new_file.write_text("new", encoding="utf-8")

        files = _collect_dump_files_since(dump_dir=tmp_path, since_epoch=since_epoch)
        assert files == [new_file]

    def test_empty_when_dir_not_exists(self, tmp_path) -> None:
        missing_dir = tmp_path / "missing"
        files = _collect_dump_files_since(dump_dir=missing_dir, since_epoch=time.time())
        assert files == []


class TestSendDiscordDumpAttachments:
    """Tests cho gửi attachments lên Discord."""

    def test_skip_when_channels_not_include_discord(self, tmp_path) -> None:
        config_reader = MagicMock()
        dump_file = tmp_path / "dump.md"
        dump_file.write_text("x", encoding="utf-8")

        _send_discord_dump_attachments(
            config_reader=config_reader,
            dump_files=[dump_file],
            channels=["telegram"],
        )

        config_reader.get_notification_settings.assert_not_called()

    @patch("chahi.infrastructure.notifiers.discord_notifier.DiscordNotifier")
    def test_send_all_files_to_enabled_discord(
        self,
        mock_discord_cls: MagicMock,
        tmp_path,
    ) -> None:
        dump1 = tmp_path / "d1.md"
        dump2 = tmp_path / "d2.md"
        dump1.write_text("1", encoding="utf-8")
        dump2.write_text("2", encoding="utf-8")

        settings = [
            NotificationSettings(
                type="discord",
                enabled=True,
                webhook_url="https://discord.com/api/webhooks/123/abc",
            ),
        ]
        config_reader = MagicMock()
        config_reader.get_notification_settings.return_value = settings

        notifier = MagicMock()
        mock_discord_cls.return_value = notifier

        _send_discord_dump_attachments(
            config_reader=config_reader,
            dump_files=[dump1, dump2],
            channels=None,
        )

        mock_discord_cls.assert_called_once()
        assert notifier.send_file_attachment.call_count == 2
