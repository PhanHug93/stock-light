"""Domain interfaces — abstract contracts giữa core và infrastructure.

Module này tập hợp toàn bộ "hợp đồng giao tiếp" (Ports) theo
Dependency Inversion Principle. Core layer chỉ phụ thuộc vào
các ABC này, không bao giờ phụ thuộc trực tiếp vào implementation.

Quy tắc:
    - Chỉ sử dụng Python built-in và abc module.
    - Import entities từ cùng layer core/.
    - Mỗi interface = một trách nhiệm duy nhất (ISP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from chahi.core.entities import (
    Article,
    LLMSettings,
    SourceCategory,
    SourceConfig,
)


class IConfigReader(ABC):
    """Contract cho component đọc cấu hình ứng dụng.

    Trừu tượng hóa nguồn cấu hình — core layer không cần biết
    config đến từ YAML, TOML, database, hay bất kỳ nguồn nào.
    """

    @abstractmethod
    def get_sources(self) -> dict[SourceCategory, list[SourceConfig]]:
        """Lấy danh sách nguồn tin, phân nhóm theo danh mục.

        Returns:
            Dictionary mapping từ SourceCategory → list SourceConfig.
            Mỗi category chứa ít nhất 1 source.

        Raises:
            FileNotFoundError: Khi nguồn cấu hình không tồn tại.
            ValueError: Khi cấu hình không hợp lệ.
        """

    @abstractmethod
    def get_llm_settings(self) -> LLMSettings:
        """Lấy cấu hình kết nối LLM.

        Returns:
            LLMSettings đã được validate và sẵn sàng sử dụng.

        Raises:
            FileNotFoundError: Khi nguồn cấu hình không tồn tại.
            ValueError: Khi cấu hình LLM không hợp lệ.
        """


class INewsFetcher(ABC):
    """Contract cho component cào tin tức từ nguồn bên ngoài.

    Chịu trách nhiệm kết nối tới URL, parse dữ liệu,
    và trả về danh sách Article đã chuẩn hóa.
    """

    @abstractmethod
    def fetch_news(self, url: str, limit: int = 10) -> list[Article]:
        """Cào tin tức từ một nguồn.

        Args:
            url: URL nguồn tin (RSS feed URL).
            limit: Số lượng bài viết tối đa cần lấy.
                   Mặc định 10. Giá trị thực tế có thể nhỏ hơn
                   nếu nguồn không có đủ bài viết.

        Returns:
            Danh sách Article đã được parse và validate.

        Raises:
            ConnectionError: Khi không thể kết nối tới URL.
            ValueError: Khi dữ liệu trả về không parse được.
        """


class ILLMClient(ABC):
    """Contract cho component giao tiếp với Large Language Model.

    Tuân theo pattern tách biệt system prompt và user content,
    phổ biến trong OpenAI-compatible API.
    """

    @abstractmethod
    def analyze(self, system_prompt: str, user_content: str) -> str:
        """Gửi prompt tới LLM và nhận phân tích.

        Args:
            system_prompt: Instructions cho LLM về vai trò và
                          format output mong muốn.
            user_content: Nội dung tin tức đã format, cần được
                         LLM phân tích.

        Returns:
            Nội dung phân tích dạng text (Markdown) từ LLM.

        Raises:
            ConnectionError: Khi không thể kết nối tới LLM API.
            RuntimeError: Khi LLM trả về response không hợp lệ.
        """


class IMemoryManager(ABC):
    """Contract cho component quản lý bộ nhớ dài hạn (Long-term Memory).

    Chịu trách nhiệm lưu trữ và truy xuất nhận định phân tích
    giữa các lần chạy, tạo feedback loop cho AI:
        Retrieve → Contextualize → Store

    Tuân thủ DI: core layer không biết memory được lưu ở đâu
    (file, MCP server, database, ...).
    """

    @abstractmethod
    def retrieve_last_context(self) -> str | None:
        """Truy xuất nhận định/tóm tắt của phiên phân tích gần nhất.

        Returns:
            Chuỗi text chứa nhận định cũ, hoặc None nếu chưa có
            (lần chạy đầu tiên).
        """

    @abstractmethod
    def save_context(self, context_data: str) -> None:
        """Lưu trữ nhận định mới sau khi LLM phân tích xong.

        Args:
            context_data: Nội dung cần lưu (tóm tắt hoặc toàn bộ report).

        Raises:
            RuntimeError: Khi không thể lưu được (disk full, MCP down, ...).
        """


class INotifier(ABC):
    """Contract cho component gửi thông báo báo cáo.

    Hỗ trợ nhiều kênh: Telegram, Discord, Slack, v.v.
    Mỗi implementation xử lý formatting và rate limiting riêng.
    """

    @abstractmethod
    def send_report(self, title: str, content: str) -> bool:
        """Gửi báo cáo tới kênh thông báo.

        Args:
            title: Tiêu đề báo cáo (vd: "ChaHi Report 2026-03-19").
            content: Nội dung báo cáo Markdown.

        Returns:
            True nếu gửi thành công, False nếu thất bại.
        """
