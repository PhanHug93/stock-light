"""Gemini LLM Client — implementation of ILLMClient cho Google Gemini.

Sử dụng ``google-genai`` SDK để kết nối với Gemini API.
Hỗ trợ cả Gemini 2.0 Flash và các model khác.
"""

from __future__ import annotations

import logging

from google import genai
from google.genai import types

from chahi.core.entities import LLMSettings
from chahi.core.interfaces import ILLMClient

logger = logging.getLogger(__name__)


class GeminiClient(ILLMClient):
    """Giao tiếp với Google Gemini API.

    Implement ILLMClient — sử dụng google-genai SDK
    với system instruction và user content riêng biệt.

    Args:
        settings: Cấu hình LLM (api_key, model_name, temperature).
    """

    _DEFAULT_TIMEOUT: int = 120  # seconds

    def __init__(
        self,
        settings: LLMSettings,
        timeout: int | None = None,
    ) -> None:
        self._settings = settings
        self._timeout = timeout or self._DEFAULT_TIMEOUT

        self._client = genai.Client(api_key=settings.api_key)

        logger.info(
            "GeminiClient khởi tạo: model=%s, temp=%.1f, timeout=%ds",
            settings.model_name,
            settings.temperature,
            self._timeout,
        )

    def analyze(self, system_prompt: str, user_content: str) -> str:
        """Gửi prompt tới Gemini và nhận phân tích.

        Args:
            system_prompt: Instructions cho LLM (system instruction).
            user_content: Nội dung tin tức cần phân tích.

        Returns:
            Nội dung phân tích Markdown từ Gemini.

        Raises:
            ConnectionError: Khi không thể kết nối tới Gemini API.
            RuntimeError: Khi response không hợp lệ hoặc API error.
        """
        logger.info(
            "Gửi request tới Gemini: model=%s, system=%d chars, user=%d chars",
            self._settings.model_name,
            len(system_prompt),
            len(user_content),
        )

        try:
            response = self._client.models.generate_content(
                model=self._settings.model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self._settings.temperature,
                    max_output_tokens=4096,
                ),
            )
        except Exception as exc:
            error_name = type(exc).__name__
            if "api_key" in str(exc).lower() or "permission" in str(exc).lower():
                msg = (
                    "Lỗi xác thực Gemini: Kiểm tra lại API key. "
                    "Lấy key tại: https://aistudio.google.com/apikey"
                )
                logger.error(msg)
                raise ConnectionError(msg) from None
            msg = f"Gemini API lỗi ({error_name}): {exc}"
            logger.error(msg)
            raise RuntimeError(msg) from None

        # ── Validate response ──
        if not response.text:
            msg = "Gemini trả về response rỗng."
            logger.error(msg)
            raise RuntimeError(msg)

        result = response.text.strip()
        logger.info("Gemini response: %d chars", len(result))
        return result
