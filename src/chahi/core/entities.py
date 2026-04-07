"""Domain entities — value objects thuần túy, không phụ thuộc thư viện ngoài.

Module này chứa toàn bộ data structures của ChaHi domain:
cấu hình nguồn tin, cấu hình LLM, bài viết, và ngữ cảnh phân tích.

Quy tắc:
    - Chỉ sử dụng Python built-in và dataclasses.
    - frozen=True + slots=True cho immutability và performance.
    - Validation tại __post_init__ — fail fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from datetime import date, datetime

# ═════════════════════════════════════════════════════════════
# Enums
# ═════════════════════════════════════════════════════════════


@unique
class SourceCategory(Enum):
    """Danh mục nguồn tin tài chính.

    Ba nhóm chính phục vụ phân tích vĩ mô:
        - OIL_MACRO: Tin dầu và kinh tế vĩ mô (gộp chung vì liên quan mật thiết).
        - GOLD: Tin vàng và kim loại quý.
        - CRYPTO: Tin tiền điện tử.
    """

    OIL_MACRO = "oil_macro"
    GOLD = "gold"
    CRYPTO = "crypto"

    def __str__(self) -> str:
        """Trả về label hiển thị thân thiện.

        Returns:
            Tên danh mục đã được format (vd: "Oil & Macro").
        """
        labels: dict[str, str] = {
            "oil_macro": "Oil & Macro",
            "gold": "Gold",
            "crypto": "Crypto",
        }
        return labels[self.value]


# ═════════════════════════════════════════════════════════════
# Config Value Objects
# ═════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """Cấu hình cho một nguồn tin cụ thể.

    Attributes:
        name: Tên hiển thị của nguồn (vd: "Reuters Business").
        url: URL để fetch dữ liệu (RSS feed URL).
        type: Phương thức lấy dữ liệu. Hiện chỉ hỗ trợ "rss".
    """

    name: str
    url: str
    type: str = "rss"

    def __post_init__(self) -> None:
        """Validate dữ liệu bắt buộc.

        Raises:
            ValueError: Khi name/url trống, type/scheme không hợp lệ.
        """
        if not self.name.strip():
            raise ValueError("SourceConfig.name không được để trống.")
        if not self.url.strip():
            raise ValueError("SourceConfig.url không được để trống.")

        # Chống SSRF: chỉ cho phép http/https
        parsed = urlparse(self.url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"SourceConfig.url phải dùng http hoặc https, "
                f"nhận được scheme: '{parsed.scheme}'. "
                f"URL: {self.url}"
            )

        if self.type not in ("rss",):
            raise ValueError(
                f"SourceConfig.type không hợp lệ: '{self.type}'. "
                f"Hiện chỉ hỗ trợ: 'rss'."
            )


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Cấu hình kết nối tới LLM provider.

    Hỗ trợ Strategy Pattern: chuyển đổi giữa LM Studio (local)
    và cloud providers (Gemini/OpenAI) qua field ``provider``.

    Attributes:
        provider: Tên provider ("lm_studio", "gemini", "openai").
        api_base: Base URL của API endpoint (LM Studio/OpenAI-compatible).
        api_key: API key cho provider.
        model_name: Tên model.
        temperature: Độ sáng tạo (0.0 = deterministic, 2.0 = max).
    """

    _VALID_PROVIDERS: tuple[str, ...] = ("lm_studio", "gemini", "openai")

    provider: str = "gemini"
    api_base: str = ""
    api_key: str = ""
    model_name: str = "gemini-3.1-pro"
    temperature: float = 0.1
    timeout: int = 120  # seconds — local LLM inference có thể chậm

    def __post_init__(self) -> None:
        """Validate giá trị hợp lệ.

        Raises:
            ValueError: Khi provider/api_base/temperature không hợp lệ.
        """
        if self.provider not in self._VALID_PROVIDERS:
            raise ValueError(
                f"LLMSettings.provider không hợp lệ: '{self.provider}'. "
                f"Hỗ trợ: {list(self._VALID_PROVIDERS)}"
            )
        if self.provider == "lm_studio" and not self.api_base.strip():
            raise ValueError("LLMSettings.api_base không được để trống cho lm_studio.")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(
                f"LLMSettings.temperature phải trong [0.0, 2.0], "
                f"nhận được: {self.temperature}"
            )
        if not (10 <= self.timeout <= 600):
            raise ValueError(
                f"LLMSettings.timeout phải trong [10, 600] giây, "
                f"nhận được: {self.timeout}"
            )


@dataclass(frozen=True, slots=True)
class MemorySettings:
    """Cấu hình Memory Manager — chọn backend lưu trữ nhận định.

    Attributes:
        type: Loại backend ("file" hoặc "mcp").
        url: URL của MCP server (chỉ dùng khi type="mcp").
        workspace_path: Đường dẫn workspace cho MCP (optional).
        protocol: Giao thức MCP ("rest" hoặc "jsonrpc").
            - rest: Custom REST (POST /call-tool).
            - jsonrpc: Chuẩn MCP SSE bridge (JSON-RPC 2.0 + SSE handshake).
    """

    _VALID_TYPES: tuple[str, ...] = ("file", "mcp")
    _VALID_PROTOCOLS: tuple[str, ...] = ("rest", "jsonrpc")

    type: str = "file"
    url: str = ""
    workspace_path: str = "."
    protocol: str = "rest"

    def __post_init__(self) -> None:
        """Validate cấu hình.

        Raises:
            ValueError: Khi type/protocol không hợp lệ hoặc thiếu URL cho MCP.
        """
        if self.type not in self._VALID_TYPES:
            raise ValueError(
                f"MemorySettings.type không hợp lệ: '{self.type}'. "
                f"Hỗ trợ: {list(self._VALID_TYPES)}"
            )
        if self.type == "mcp" and not self.url.strip():
            raise ValueError("MemorySettings.url không được để trống khi type='mcp'.")
        if self.type == "mcp" and self.protocol not in self._VALID_PROTOCOLS:
            raise ValueError(
                f"MemorySettings.protocol không hợp lệ: '{self.protocol}'. "
                f"Hỗ trợ: {list(self._VALID_PROTOCOLS)}"
            )


@dataclass(frozen=True, slots=True)
class NotificationSettings:
    """Cấu hình gửi thông báo báo cáo qua Telegram/Discord.

    Attributes:
        type: Loại kênh ("telegram" hoặc "discord").
        enabled: Bật/tắt kênh này.
        bot_token: Bot token (Telegram only, từ @BotFather).
        chat_id: Chat/Group ID (Telegram only).
        webhook_url: Webhook URL (Discord only).
    """

    _VALID_TYPES: tuple[str, ...] = ("telegram", "discord")

    type: str = "telegram"
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""
    webhook_url: str = ""

    def __post_init__(self) -> None:
        """Validate cấu hình.

        Raises:
            ValueError: Khi type không hợp lệ hoặc thiếu thông tin bắt buộc.
        """
        if self.type not in self._VALID_TYPES:
            raise ValueError(
                f"NotificationSettings.type không hợp lệ: '{self.type}'. "
                f"Hỗ trợ: {list(self._VALID_TYPES)}"
            )
        if self.enabled and self.type == "telegram":
            if not self.bot_token.strip():
                raise ValueError(
                    "NotificationSettings: bot_token bắt buộc cho Telegram."
                )
            if not self.chat_id.strip():
                raise ValueError("NotificationSettings: chat_id bắt buộc cho Telegram.")
        if self.enabled and self.type == "discord" and not self.webhook_url.strip():
            raise ValueError("NotificationSettings: webhook_url bắt buộc cho Discord.")


# ═════════════════════════════════════════════════════════════
# Domain Entities
# ═════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class Article:
    """Một bài viết tin tức tài chính — immutable value object.

    Đây là đơn vị dữ liệu cơ bản nhất, đại diện cho một mẩu tin
    đã được fetch và parse từ nguồn RSS.

    Attributes:
        title: Tiêu đề bài viết.
        summary: Tóm tắt nội dung (text, đã strip HTML nếu có).
        source_name: Tên nguồn tin (vd: "Reuters", "CoinDesk").
        published_date: Thời điểm xuất bản.
        url: Đường dẫn gốc tới bài viết.
    """

    title: str
    summary: str
    source_name: str
    published_date: datetime
    url: str

    def __post_init__(self) -> None:
        """Validate dữ liệu bắt buộc.

        Raises:
            ValueError: Khi title hoặc url trống.
        """
        if not self.title.strip():
            raise ValueError("Article.title không được để trống.")
        if not self.url.strip():
            raise ValueError("Article.url không được để trống.")


@dataclass(frozen=True, slots=True)
class AnalysisContext:
    """Ngữ cảnh phân tích — gom tin tức theo danh mục cho một ngày.

    Đây là input đầu vào cho LLM: tổng hợp toàn bộ tin tức
    đã fetch được, phân nhóm theo danh mục.

    Sử dụng ``news_by_category`` (dict) thay vì hard-code từng field
    để tuân thủ OCP — thêm category mới chỉ cần thêm Enum value.

    Attributes:
        date: Ngày phân tích.
        news_by_category: Tin tức phân nhóm theo SourceCategory.
        previous_context: Nhận định của ngày hôm trước (từ Memory).
    """

    date: date
    news_by_category: dict[SourceCategory, list[Article]] = field(
        default_factory=dict,
    )
    previous_context: str | None = None

    # ── Backward-compat: giữ cho test cũ và code chưa migrate ──

    @staticmethod
    def _compat_init(
        *,
        date_val: date,
        oil_news: list[Article] | None = None,
        gold_news: list[Article] | None = None,
        crypto_news: list[Article] | None = None,
        previous_context: str | None = None,
        news_by_category: dict[SourceCategory, list[Article]] | None = None,
    ) -> AnalysisContext:
        """Factory tạo AnalysisContext từ old-style kwargs hoặc dict.

        Ưu tiên ``news_by_category`` nếu có; ngược lại build từ 3 list.
        """
        if news_by_category is not None:
            return AnalysisContext(
                date=date_val,
                news_by_category=news_by_category,
                previous_context=previous_context,
            )
        cat_map: dict[SourceCategory, list[Article]] = {}
        if oil_news:
            cat_map[SourceCategory.OIL_MACRO] = oil_news
        if gold_news:
            cat_map[SourceCategory.GOLD] = gold_news
        if crypto_news:
            cat_map[SourceCategory.CRYPTO] = crypto_news
        return AnalysisContext(
            date=date_val,
            news_by_category=cat_map,
            previous_context=previous_context,
        )

    @property
    def oil_news(self) -> list[Article]:
        """Backward-compat property."""
        return self.news_by_category.get(SourceCategory.OIL_MACRO, [])

    @property
    def gold_news(self) -> list[Article]:
        """Backward-compat property."""
        return self.news_by_category.get(SourceCategory.GOLD, [])

    @property
    def crypto_news(self) -> list[Article]:
        """Backward-compat property."""
        return self.news_by_category.get(SourceCategory.CRYPTO, [])

    @property
    def total_articles(self) -> int:
        """Tổng số bài viết trên tất cả danh mục.

        Returns:
            Tổng số Article trong mọi category.
        """
        return sum(len(articles) for articles in self.news_by_category.values())

    @property
    def is_empty(self) -> bool:
        """Kiểm tra có bài viết nào không.

        Returns:
            True nếu không có bài viết nào ở bất kỳ danh mục nào.
        """
        return self.total_articles == 0
