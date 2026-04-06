"""OpenAI Client — implementation of ILLMClient cho OpenAI API."""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

import openai

from chahi.core.interfaces import ILLMClient

if TYPE_CHECKING:
    from chahi.core.entities import LLMSettings

logger = logging.getLogger(__name__)


class OpenAIClient(ILLMClient):
    """Client giao tiếp trực tiếp với OpenAI Chat Completions API.

    Hỗ trợ custom ``api_base`` để tương thích gateway/proxy.
    Nếu ``api_base`` để trống sẽ dùng endpoint mặc định của OpenAI.
    """

    _DEFAULT_API_BASE: str = "https://api.openai.com/v1"
    _MAX_RETRIES: int = 2
    _RETRY_BASE_DELAY: float = 2.0  # seconds

    def __init__(
        self,
        settings: LLMSettings,
    ) -> None:
        self._settings = settings
        self._timeout = settings.timeout

        api_key = settings.api_key.strip() or os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            msg = (
                "Thiếu OpenAI API key: truyền `llm_settings.api_key`, "
                "hoặc export biến môi trường OPENAI_API_KEY."
            )
            logger.error(msg)
            raise ValueError(msg)

        api_base = settings.api_base.strip() or self._DEFAULT_API_BASE

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": api_base,
            "timeout": self._timeout,
        }
        self._client = openai.OpenAI(**client_kwargs)

        logger.info(
            "OpenAIClient khởi tạo: base=%s, model=%s, temp=%.1f, timeout=%ds",
            api_base,
            settings.model_name,
            settings.temperature,
            self._timeout,
        )

    @property
    def supports_concurrency(self) -> bool:
        """OpenAI self-serve thường có rate-limit chặt, ưu tiên tuần tự để ổn định."""
        return False

    def analyze(self, system_prompt: str, user_content: str) -> str:
        """Gửi prompt tới OpenAI và nhận phân tích."""
        logger.info(
            "Gửi request tới OpenAI: model=%s, system=%d chars, user=%d chars",
            self._settings.model_name,
            len(system_prompt),
            len(user_content),
        )

        total_attempts = self._MAX_RETRIES + 1
        response = None

        for attempt in range(1, total_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._settings.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=self._settings.temperature,
                )
                break
            except openai.APITimeoutError:
                msg = (
                    f"OpenAI timeout sau {self._timeout}s. "
                    f"Hãy giảm độ dài input hoặc tăng timeout."
                )
                logger.error(msg)
                raise RuntimeError(msg) from None
            except openai.AuthenticationError:
                msg = "Lỗi xác thực OpenAI: kiểm tra lại OPENAI_API_KEY hoặc api_key."
                logger.error(msg)
                raise ConnectionError(msg) from None
            except openai.APIConnectionError:
                msg = (
                    "Không thể kết nối OpenAI API. "
                    "Kiểm tra mạng hoặc api_base cấu hình."
                )
                logger.error(msg)
                raise ConnectionError(msg) from None
            except openai.RateLimitError as exc:
                error_code, error_message = self._extract_error_details(exc)
                if error_code in {"insufficient_quota", "billing_hard_limit_reached"}:
                    msg = (
                        "OpenAI API báo hết quota/billing limit. "
                        "Lưu ý ChatGPT Plus không bao gồm API credits; "
                        "cần nạp quota tại https://platform.openai.com/billing. "
                        f"Chi tiết: {error_message or 'insufficient_quota'}"
                    )
                    logger.error(msg)
                    raise RuntimeError(msg) from None

                if attempt >= total_attempts:
                    msg = (
                        "OpenAI API rate-limit liên tục (HTTP 429). "
                        "Thử lại sau vài phút hoặc giảm tần suất request."
                    )
                    logger.error(msg)
                    raise RuntimeError(msg) from None

                delay = self._RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "OpenAI rate-limit (attempt %d/%d), retry sau %.1fs...",
                    attempt,
                    total_attempts,
                    delay,
                )
                time.sleep(delay)
            except openai.APIStatusError as exc:
                msg = f"OpenAI API lỗi HTTP {exc.status_code}: {exc.message}"
                logger.error(msg)
                raise RuntimeError(msg) from None

        if response is None:
            msg = "OpenAI không trả về response hợp lệ."
            logger.error(msg)
            raise RuntimeError(msg)

        if not response.choices:
            msg = "OpenAI trả về response không có choices."
            logger.error(msg)
            raise RuntimeError(msg)

        content = response.choices[0].message.content
        if not content or not content.strip():
            msg = "OpenAI trả về nội dung rỗng."
            logger.error(msg)
            raise RuntimeError(msg)

        result = content.strip()
        logger.info(
            "OpenAI response: %d chars, finish_reason=%s",
            len(result),
            response.choices[0].finish_reason,
        )
        return result

    @staticmethod
    def _extract_error_details(exc: openai.RateLimitError) -> tuple[str, str]:
        """Trích xuất (error_code, error_message) từ RateLimitError."""
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error_obj = body.get("error")
            if isinstance(error_obj, dict):
                code = str(error_obj.get("code") or "")
                message = str(error_obj.get("message") or "")
                return code, message
        return "", str(exc)
