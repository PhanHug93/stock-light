"""LM Studio Client — implementation of ILLMClient.

Kết nối tới LM Studio local server thông qua OpenAI-compatible API.
Sử dụng thư viện ``openai`` Python chính thức, chỉ cần override ``base_url``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import openai

from chahi.core.interfaces import ILLMClient

if TYPE_CHECKING:
    from chahi.core.entities import LLMSettings

logger = logging.getLogger(__name__)


class LMStudioClient(ILLMClient):
    """Client giao tiếp với LM Studio qua OpenAI-compatible API.

    LM Studio chạy local inference server tại ``http://localhost:1234/v1``
    và expose API tương thích hoàn toàn với OpenAI Chat Completions.

    Args:
        settings: Cấu hình kết nối LLM (api_base, api_key, model_name,
                  temperature, timeout).
    """

    def __init__(
        self,
        settings: LLMSettings,
    ) -> None:
        self._settings = settings
        self._timeout = settings.timeout
        self._client = openai.OpenAI(
            base_url=settings.api_base,
            api_key=settings.api_key,
            timeout=self._timeout,
        )
        logger.info(
            "LMStudioClient khởi tạo: base=%s, model=%s, temp=%.1f, timeout=%ds",
            settings.api_base,
            settings.model_name,
            settings.temperature,
            self._timeout,
        )

    def analyze(self, system_prompt: str, user_content: str) -> str:
        """Gửi prompt tới LM Studio và nhận phân tích.

        Sử dụng Chat Completions API với 2 messages:
            - ``system``: instructions về vai trò và format output.
            - ``user``: nội dung tin tức cần phân tích.

        Args:
            system_prompt: Instructions cho LLM (role="system").
            user_content: Nội dung tin tức đã format (role="user").

        Returns:
            Nội dung phân tích dạng text (Markdown) từ LLM.

        Raises:
            ConnectionError: Khi LM Studio server chưa khởi động.
            RuntimeError: Khi response từ LLM rỗng hoặc không hợp lệ.
        """
        logger.info(
            "Gửi request tới LLM: model=%s, "
            "system_prompt=%d chars, user_content=%d chars",
            self._settings.model_name,
            len(system_prompt),
            len(user_content),
        )

        try:
            response = self._client.chat.completions.create(
                model=self._settings.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=self._settings.temperature,
            )
        except openai.APITimeoutError:
            msg = (
                f"LLM timeout sau {self._timeout}s. "
                f"Model có thể quá lớn hoặc input quá dài."
            )
            logger.error(msg)
            raise RuntimeError(msg) from None
        except openai.APIConnectionError:
            msg = (
                "Lỗi kết nối: Vui lòng kiểm tra xem bạn đã "
                "bấm Start Server trên LM Studio chưa."
            )
            logger.error(msg)
            raise ConnectionError(msg) from None
        except openai.RateLimitError:
            msg = "LLM server quá tải (rate limit). Thử lại sau."
            logger.error(msg)
            raise RuntimeError(msg) from None
        except openai.APIStatusError as exc:
            msg = f"LLM API lỗi HTTP {exc.status_code}: {exc.message}"
            logger.error(msg)
            raise RuntimeError(msg) from None

        # ── Validate response ──
        if not response.choices:
            msg = "LLM trả về response không có choices."
            logger.error(msg)
            raise RuntimeError(msg)

        content = response.choices[0].message.content

        if not content or not content.strip():
            msg = "LLM trả về nội dung rỗng."
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info(
            "Nhận response thành công: %d chars, finish_reason=%s",
            len(content),
            response.choices[0].finish_reason,
        )

        return content.strip()
