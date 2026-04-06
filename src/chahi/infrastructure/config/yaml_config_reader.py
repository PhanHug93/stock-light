"""YamlConfigReader — implementation of IConfigReader.

Đọc file YAML config, parse và map sang domain entities.
Xử lý triệt để exception: file missing, YAML syntax error, missing fields.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import yaml

from chahi.core.entities import (
    LLMSettings,
    MemorySettings,
    NotificationSettings,
    SourceCategory,
    SourceConfig,
)
from chahi.core.interfaces import IConfigReader

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "gemini": {
        "api_base": "",
        "api_key": "",
        "model_name": "gemini-2.0-flash",
    },
    "lm_studio": {
        "api_base": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "model_name": "default",
    },
    "openai": {
        "api_base": "https://api.openai.com/v1",
        "api_key": "",
        "model_name": "gpt-5.4",
    },
}


class YamlConfigReader(IConfigReader):
    """Đọc cấu hình ứng dụng từ file YAML.

    Implement IConfigReader — chịu trách nhiệm:
    1. Đọc và parse file YAML.
    2. Map section ``sources`` sang dict[SourceCategory, list[SourceConfig]].
    3. Map section ``llm_settings`` sang LLMSettings.
    4. Raise exception rõ ràng khi có lỗi.

    Args:
        config_path: Đường dẫn tới file YAML cấu hình.
    """

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path
        self._raw: dict[str, Any] | None = None

    def get_sources(self) -> dict[SourceCategory, list[SourceConfig]]:
        """Lấy danh sách nguồn tin phân nhóm theo danh mục.

        Returns:
            Dictionary mapping SourceCategory → list[SourceConfig].

        Raises:
            FileNotFoundError: Khi file config không tồn tại.
            ValueError: Khi section ``sources`` thiếu, rỗng, hoặc sai format.
            yaml.YAMLError: Khi file YAML có lỗi cú pháp.
        """
        raw = self._load_raw()
        sources_raw = raw.get("sources")

        if not sources_raw or not isinstance(sources_raw, dict):
            raise ValueError(
                "Config phải có key 'sources' dạng dictionary "
                "với các category (oil_macro, gold, crypto)."
            )

        result: dict[SourceCategory, list[SourceConfig]] = {}

        for category_key, feeds_list in sources_raw.items():
            category = self._parse_category(category_key)
            sources = self._parse_source_list(feeds_list, category_key)
            result[category] = sources
            logger.info(
                "Loaded %d sources cho [%s]",
                len(sources),
                category,
            )

        if not result:
            raise ValueError("Config sources không có category nào.")

        return result

    def get_llm_settings(self) -> LLMSettings:
        """Lấy cấu hình kết nối LLM.

        Nếu section ``llm_settings`` không tồn tại, trả về giá trị mặc định.

        Returns:
            LLMSettings đã validate.

        Raises:
            FileNotFoundError: Khi file config không tồn tại.
            ValueError: Khi giá trị cấu hình không hợp lệ.
            yaml.YAMLError: Khi file YAML có lỗi cú pháp.
        """
        raw = self._load_raw()
        llm_raw = raw.get("llm_settings", {})

        if not isinstance(llm_raw, dict):
            raise ValueError(
                f"Config 'llm_settings' phải là dictionary, "
                f"nhận được: {type(llm_raw).__name__}"
            )

        provider = str(llm_raw.get("provider", "gemini")).strip().lower()
        defaults = _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["gemini"])

        settings = LLMSettings(
            provider=provider,
            api_base=str(llm_raw.get("api_base", defaults["api_base"])),
            api_key=str(llm_raw.get("api_key", defaults["api_key"])),
            model_name=str(llm_raw.get("model_name", defaults["model_name"])),
            temperature=float(llm_raw.get("temperature", 0.1)),
            timeout=int(llm_raw.get("timeout", 120)),
        )

        masked_key = (
            settings.api_key[:3] + "****" if len(settings.api_key) > 3 else "****"
        )
        logger.info(
            "LLM settings: provider=%s, base=%s, model=%s, key=%s, temp=%.1f",
            settings.provider,
            settings.api_base,
            settings.model_name,
            masked_key,
            settings.temperature,
        )

        return settings

    def get_memory_settings(self) -> MemorySettings:
        """Lấy cấu hình Memory Manager.

        Returns:
            MemorySettings đã validate.
        """
        raw = self._load_raw()
        mem_raw = raw.get("memory", {})

        if not isinstance(mem_raw, dict):
            return MemorySettings()  # defaults

        settings = MemorySettings(
            type=str(mem_raw.get("type", "file")),
            url=str(mem_raw.get("url", "")),
            workspace_path=str(mem_raw.get("workspace_path", ".")),
            protocol=str(mem_raw.get("protocol", "rest")),
        )

        logger.info(
            "Memory settings: type=%s, protocol=%s, url=%s",
            settings.type,
            settings.protocol,
            settings.url or "(local)",
        )

        return settings

    def get_notification_settings(self) -> list[NotificationSettings]:
        """Lấy cấu hình thông báo (Telegram, Discord).

        Đọc section ``notifications`` trong YAML config và trả về
        danh sách NotificationSettings cho các kênh được cấu hình.

        Returns:
            Danh sách NotificationSettings (có thể rỗng nếu không cấu hình).
        """
        raw = self._load_raw()
        notif_raw = raw.get("notifications", {})

        if not isinstance(notif_raw, dict):
            return []

        result: list[NotificationSettings] = []

        # ── Telegram ──
        tg_raw = notif_raw.get("telegram", {})
        if isinstance(tg_raw, dict):
            try:
                result.append(
                    NotificationSettings(
                        type="telegram",
                        enabled=bool(tg_raw.get("enabled", False)),
                        bot_token=str(tg_raw.get("bot_token", "")),
                        chat_id=str(tg_raw.get("chat_id", "")),
                    )
                )
            except ValueError as exc:
                logger.warning("Telegram notification config lỗi: %s", exc)

        # ── Discord ──
        dc_raw = notif_raw.get("discord", {})
        if isinstance(dc_raw, dict):
            try:
                result.append(
                    NotificationSettings(
                        type="discord",
                        enabled=bool(dc_raw.get("enabled", False)),
                        webhook_url=str(dc_raw.get("webhook_url", "")),
                    )
                )
            except ValueError as exc:
                logger.warning("Discord notification config lỗi: %s", exc)

        enabled_count = sum(1 for s in result if s.enabled)
        logger.info("Notification settings: %d kênh enabled", enabled_count)

        return result

    # ── Private helpers ─────────────────────────────────────────

    def _load_raw(self) -> dict[str, Any]:
        """Đọc và cache raw YAML data.

        Sử dụng lazy loading + cache: chỉ đọc file 1 lần,
        các lần gọi sau dùng cache.

        Returns:
            Dictionary chứa raw data từ YAML.

        Raises:
            FileNotFoundError: Khi file không tồn tại.
            ValueError: Khi file rỗng hoặc không phải dict.
            yaml.YAMLError: Khi YAML có lỗi cú pháp.
        """
        if self._raw is not None:
            return self._raw

        if not self._config_path.exists():
            raise FileNotFoundError(f"Config file không tồn tại: {self._config_path}")

        logger.info("Đọc config từ: %s", self._config_path)

        try:
            with self._config_path.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise yaml.YAMLError(
                f"Lỗi cú pháp YAML trong {self._config_path}: {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise ValueError(
                f"Config file phải là YAML dictionary, nhận được: {type(raw).__name__}"
            )

        self._raw = raw
        return raw

    @staticmethod
    def _parse_category(key: str) -> SourceCategory:
        """Parse category string thành SourceCategory enum.

        Args:
            key: Category key từ YAML (vd: "oil_macro", "gold", "crypto").

        Returns:
            SourceCategory tương ứng.

        Raises:
            ValueError: Khi key không khớp với bất kỳ category nào.
        """
        try:
            return SourceCategory(key.lower().strip())
        except ValueError:
            valid = [c.value for c in SourceCategory]
            raise ValueError(
                f"Category '{key}' không hợp lệ. Các giá trị hợp lệ: {valid}"
            ) from None

    @staticmethod
    def _parse_source_list(
        feeds_list: Any,
        category_key: str,
    ) -> list[SourceConfig]:
        """Parse danh sách sources cho một category.

        Args:
            feeds_list: Raw list từ YAML.
            category_key: Tên category (dùng cho error message).

        Returns:
            Danh sách SourceConfig đã validate.

        Raises:
            ValueError: Khi feeds_list không hợp lệ.
        """
        if not isinstance(feeds_list, list) or not feeds_list:
            raise ValueError(
                f"Category '{category_key}' phải chứa list sources, không được rỗng."
            )

        result: list[SourceConfig] = []

        for idx, item in enumerate(feeds_list):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Source #{idx} trong '{category_key}' phải là dictionary, "
                    f"nhận được: {type(item).__name__}"
                )

            name = item.get("name")
            url = item.get("url")
            source_type = item.get("type", "rss")

            if not name:
                raise ValueError(f"Source #{idx} trong '{category_key}': thiếu 'name'.")
            if not url:
                raise ValueError(f"Source #{idx} trong '{category_key}': thiếu 'url'.")

            result.append(
                SourceConfig(name=str(name), url=str(url), type=str(source_type))
            )

        return result
