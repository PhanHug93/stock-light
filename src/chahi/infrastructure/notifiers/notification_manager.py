"""NotificationManager — Composite Pattern cho hệ thống thông báo.

Gom nhiều INotifier (Telegram, Discord, ...) thành một interface duy nhất.
Đảm bảo fail-safe: một nền tảng lỗi không ảnh hưởng nền tảng khác.
"""

from __future__ import annotations

import logging

from chahi.core.interfaces import INotifier

logger = logging.getLogger(__name__)


class NotificationManager(INotifier):
    """Composite Notifier — gửi báo cáo tới nhiều kênh cùng lúc.

    Implement INotifier theo Composite Pattern:
    - Nhận danh sách ``INotifier`` clients đã enabled.
    - ``send_report()`` vòng lặp qua tất cả clients.
    - Bắt exception độc lập cho từng client (try/except riêng).
    - Một nền tảng lỗi mạng KHÔNG làm sập nền tảng khác.

    Args:
        clients: Danh sách INotifier đã khởi tạo và enabled.
    """

    def __init__(self, clients: list[INotifier]) -> None:
        self._clients = clients
        logger.info(
            "NotificationManager: %d kênh đã đăng ký",
            len(clients),
        )

    @property
    def client_count(self) -> int:
        """Số lượng client đã đăng ký.

        Returns:
            Số lượng INotifier trong composite.
        """
        return len(self._clients)

    def send_report(self, title: str, content: str) -> bool:
        """Gửi báo cáo tới tất cả kênh thông báo đã đăng ký.

        Mỗi kênh được gọi trong try/except riêng biệt:
        - Nền tảng A lỗi → vẫn tiếp tục gửi nền tảng B.
        - Chỉ return True khi TẤT CẢ kênh đều thành công.

        Args:
            title: Tiêu đề báo cáo (vd: "ChaHi Report 2026-03-19").
            content: Nội dung báo cáo Markdown.

        Returns:
            True nếu tất cả kênh gửi thành công.
            False nếu bất kỳ kênh nào thất bại (nhưng vẫn thử hết).
        """
        if not self._clients:
            logger.info("NotificationManager: không có kênh nào được bật.")
            return True  # Không có kênh = không có lỗi

        all_success = True

        for client in self._clients:
            client_name = type(client).__name__
            try:
                success = client.send_report(title, content)
                if success:
                    logger.info("✓ Đã gửi báo cáo qua %s", client_name)
                else:
                    logger.warning("✗ Gửi thất bại qua %s", client_name)
                    all_success = False
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "✗ Exception khi gửi qua %s: %s",
                    client_name,
                    exc,
                )
                all_success = False

        return all_success
