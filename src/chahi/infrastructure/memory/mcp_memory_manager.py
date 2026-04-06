"""MCP HTTP Memory Manager — implementation of IMemoryManager via HTTP/SSE.

Giao tiếp với MCP Server qua REST API hoặc JSON-RPC SSE bridge.

Hỗ trợ 2 protocol:
    - ``rest``: Custom REST format (POST /call-tool).
    - ``jsonrpc``: Chuẩn MCP SSE bridge (JSON-RPC 2.0 + SSE handshake).

Nếu MCP Server không khả dụng, trả về None (retrieve) hoặc
log warning (save) — ChaHi vẫn chạy bình thường.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import requests

from chahi.core.interfaces import IMemoryManager

if TYPE_CHECKING:
    from chahi.core.entities import MemorySettings

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT: int = 10  # seconds
_SSE_CONNECT_TIMEOUT: int = 5  # seconds


class MCPHttpMemoryManager(IMemoryManager):
    """Quản lý bộ nhớ dài hạn qua MCP Server (HTTP/SSE).

    Hỗ trợ 2 protocol:
        - ``rest``: POST /call-tool + custom JSON payload.
        - ``jsonrpc``: SSE handshake (GET /sse) + POST /message
          với JSON-RPC 2.0 format.

    Args:
        settings: Cấu hình Memory (url, workspace_path, protocol).
    """

    def __init__(self, settings: MemorySettings) -> None:
        self._base_url = settings.url.rstrip("/")
        self._workspace_path = settings.workspace_path
        self._protocol = settings.protocol
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"

        # JSON-RPC state
        self._jsonrpc_id: int = 0
        self._message_endpoint: str | None = None

        logger.info(
            "MCPHttpMemoryManager: server=%s, workspace=%s, protocol=%s",
            self._base_url,
            self._workspace_path,
            self._protocol,
        )

    def close(self) -> None:
        """Đóng HTTP session, giải phóng socket connections."""
        self._session.close()
        logger.debug("MCPHttpMemoryManager session closed.")

    def retrieve_last_context(self) -> str | None:
        """Truy xuất nhận định phiên gần nhất từ MCP server.

        Dùng temporal query thử lùi 7 ngày (gần nhất trước) để tránh:
        - RAG anti-pattern (semantic search sai thời gian)
        - Weekend Blindspot (thứ 2 query "hôm qua" = Chủ Nhật → rỗng)
        - RAG Temporal Illusion (ChromaDB trả nearest neighbor dù sai ngày)

        Sau khi MCP trả kết quả, **kiểm tra date marker** trong text
        để đảm bảo kết quả thực sự thuộc ngày đang tìm.

        Returns:
            Nhận định cũ dạng text, hoặc None nếu không có/lỗi.
        """
        _MAX_LOOKBACK_DAYS = 7  # noqa: N806

        now = datetime.now(tz=UTC)
        for days_ago in range(1, _MAX_LOOKBACK_DAYS + 1):
            target_date = (now - timedelta(days=days_ago)).strftime("%Y-%m-%d")

            # ── search_memory với ngày cụ thể ──
            result = self._call_tool(
                tool_name="search_memory",
                arguments={
                    "query": f"nhận định phân tích vĩ mô ChaHi ngày {target_date}",
                    "workspace_path": self._workspace_path,
                    "n_results": 1,
                },
            )

            if result is not None:
                text = self._extract_text_from_result(result)
                if text and self._verify_date_in_text(text, target_date):
                    logger.info(
                        "  Retrieve từ search_memory (ngày %s, -%dd): %d chars",
                        target_date,
                        days_ago,
                        len(text),
                    )
                    return text
                if text:
                    logger.debug(
                        "  MCP trả kết quả nhưng KHÔNG chứa ngày %s "
                        "(ChromaDB nearest neighbor), bỏ qua.",
                        target_date,
                    )

        # ── Fallback cuối: auto_recall không chỉ định ngày ──
        result = self._call_tool(
            tool_name="auto_recall",
            arguments={
                "user_message": "báo cáo phân tích vĩ mô gần nhất ChaHi",
                "workspace_path": self._workspace_path,
                "n_results": 1,
            },
        )

        if result is not None:
            text = self._extract_text_from_result(result)
            if text:
                logger.info("  Retrieve từ auto_recall (fallback): %d chars", len(text))
                return text

        logger.info("  Không tìm thấy nhận định cũ trong MCP.")
        return None

    def retrieve_related_context(
        self,
        hot_keywords: list[str],
        max_results: int = 3,
    ) -> str | None:
        """Truy xuất context liên quan theo hot keywords (semantic search)."""
        cleaned_keywords = [
            keyword.strip().lower() for keyword in hot_keywords if keyword.strip()
        ]
        if not cleaned_keywords:
            return None

        unique_keywords = list(dict.fromkeys(cleaned_keywords))[:10]
        query = "bài học kinh nghiệm phân tích vĩ mô liên quan tới: " + ", ".join(
            unique_keywords
        )
        result = self._call_tool(
            tool_name="search_memory",
            arguments={
                "query": query,
                "workspace_path": self._workspace_path,
                "n_results": max(1, max_results),
            },
        )

        if result is None:
            logger.info("  Semantic retrieve: không có dữ liệu cho hot keywords.")
            return None

        snippets = self._extract_text_list_from_result(result, limit=max_results)
        if not snippets:
            logger.info("  Semantic retrieve: MCP trả về rỗng cho hot keywords.")
            return None

        merged = "\n\n---\n\n".join(snippets)
        logger.info(
            "  Semantic retrieve theo keywords (%s): %d đoạn, %d chars",
            ", ".join(unique_keywords),
            len(snippets),
            len(merged),
        )
        return merged

    @staticmethod
    def _verify_date_in_text(text: str, target_date: str) -> bool:
        """Kiểm tra text có thực sự chứa ngày target_date không.

        Chống lại RAG Temporal Illusion: ChromaDB luôn trả nearest
        neighbor, nên cần xác minh ngày trong kết quả.

        Kiểm tra 3 dạng:
        - ``[ChaHi Report YYYY-MM-DD]`` (marker chính thức)
        - ``chahi_macro_report_YYYY-MM-DD`` (metadata_source)
        - ``YYYY-MM-DD`` (date xuất hiện bất kỳ đâu)
        """
        return target_date in text

    def save_context(self, context_data: str) -> None:
        """Lưu nhận định mới vào MCP server.

        Gọi tool ``store_working_context`` để lưu vào L1 (per-workspace).
        Metadata_source chứa ngày ISO để hỗ trợ temporal retrieval.
        Embed ``[ChaHi Report YYYY-MM-DD]`` marker trong text để
        chống RAG Temporal Illusion khi retrieve.

        Args:
            context_data: Nội dung nhận định cần lưu.
        """
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")

        # Embed date marker để retrieve có thể validate
        date_marker = f"[ChaHi Report {today}]"
        stamped_data = f"{date_marker}\n\n{context_data}"

        result = self._call_tool(
            tool_name="store_working_context",
            arguments={
                "text_data": stamped_data,
                "metadata_source": f"chahi_macro_report_{today}",
                "workspace_path": self._workspace_path,
                "tech_stack": "general",
            },
        )

        if result is not None:
            logger.info("  Đã lưu %d chars vào MCP (L1).", len(context_data))
        else:
            logger.warning(
                "  Không thể lưu vào MCP. Dữ liệu (%d chars) bị mất.",
                len(context_data),
            )

    # ── Private: HTTP call (dual protocol) ──────────────────

    def _call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Gọi một MCP tool qua HTTP — tự chọn protocol.

        Args:
            tool_name: Tên tool MCP (vd: "search_memory").
            arguments: Arguments dict cho tool.

        Returns:
            Response dict nếu thành công, None nếu lỗi.
        """
        if self._protocol == "jsonrpc":
            return self._call_tool_jsonrpc(tool_name, arguments)
        return self._call_tool_rest(tool_name, arguments)

    def _call_tool_rest(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Gọi MCP tool qua custom REST format.

        Format: POST /call-tool
        Payload: {"tool_name": X, "arguments": Y}
        """
        url = f"{self._base_url}/call-tool"
        payload = {
            "tool_name": tool_name,
            "arguments": arguments,
        }
        return self._http_post(url, payload, tool_name)

    def _call_tool_jsonrpc(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Gọi MCP tool qua JSON-RPC 2.0 (chuẩn MCP SSE bridge).

        Flow đúng chuẩn MCP SSE bridge:
        1. GET /sse → mở SSE stream, nhận endpoint URL.
        2. POST endpoint với JSON-RPC 2.0 payload → nhận 202 Accepted.
        3. Đọc SSE stream → nhận result event chứa JSON-RPC response.

        Fallback: Nếu POST trả JSON trực tiếp (non-SSE server) → dùng luôn.
        """
        try:
            # ── Mở SSE stream ──
            sse_url = f"{self._base_url}/sse"
            sse_response = self._session.get(
                sse_url,
                stream=True,
                timeout=_SSE_CONNECT_TIMEOUT,
            )
            sse_response.raise_for_status()

            # ── Parse SSE events để tìm endpoint ──
            message_endpoint: str | None = None
            for line in sse_response.iter_lines(decode_unicode=True):
                if line and line.startswith("data:"):
                    data = line[5:].strip()
                    if data.startswith("/") or data.startswith("http"):
                        if data.startswith("/"):
                            message_endpoint = f"{self._base_url}{data}"
                        else:
                            message_endpoint = data
                        break

            if message_endpoint is None:
                logger.warning("SSE handshake: không tìm thấy endpoint.")
                sse_response.close()
                return None

            logger.debug("  SSE endpoint: %s", message_endpoint)

            # ── POST JSON-RPC payload ──
            self._jsonrpc_id += 1
            payload: dict[str, Any] = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments,
                },
                "id": self._jsonrpc_id,
            }

            post_response = self._session.post(
                message_endpoint,
                json=payload,
                timeout=_REQUEST_TIMEOUT,
            )
            post_response.raise_for_status()

            # ── Trường hợp 1: Server trả JSON trực tiếp (non-SSE) ──
            response_text = post_response.text.strip()
            if response_text:
                try:
                    direct_result: dict[str, Any] = json.loads(response_text)
                    sse_response.close()
                    if "error" in direct_result:
                        logger.warning(
                            "MCP JSON-RPC error: tool=%s, error=%s",
                            tool_name,
                            direct_result["error"],
                        )
                        return None
                    return direct_result.get("result", direct_result)
                except json.JSONDecodeError:
                    pass  # Không phải JSON, tiếp tục đọc SSE

            # ── Trường hợp 2: 202 Accepted → Đọc result từ SSE stream ──
            logger.debug("  POST accepted, đọc result từ SSE stream...")
            event_type: str = ""
            for line in sse_response.iter_lines(decode_unicode=True):
                if line is None:
                    continue
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data = line[5:].strip()
                    if event_type == "message" and data:
                        try:
                            sse_result: dict[str, Any] = json.loads(data)
                            # Kiểm tra id khớp
                            if sse_result.get("id") == self._jsonrpc_id:
                                sse_response.close()
                                if "error" in sse_result:
                                    logger.warning(
                                        "MCP JSON-RPC error: tool=%s, error=%s",
                                        tool_name,
                                        sse_result["error"],
                                    )
                                    return None
                                return sse_result.get("result", sse_result)
                        except json.JSONDecodeError:
                            continue

            sse_response.close()
            logger.warning(
                "SSE stream kết thúc mà không nhận được result cho tool=%s",
                tool_name,
            )
            return None

        except requests.Timeout:
            logger.warning(
                "MCP timeout: tool=%s (SSE connect %ds, request %ds)",
                tool_name,
                _SSE_CONNECT_TIMEOUT,
                _REQUEST_TIMEOUT,
            )
        except requests.ConnectionError:
            logger.warning(
                "Không thể kết nối MCP server: %s. Server có đang chạy?",
                self._base_url,
            )
        except requests.RequestException as exc:
            logger.warning("MCP JSON-RPC lỗi: tool=%s, error=%s", tool_name, exc)

        return None

    def _http_post(
        self,
        url: str,
        payload: dict[str, Any],
        tool_name: str,
    ) -> dict[str, Any] | None:
        """HTTP POST helper dùng cho REST protocol.

        Args:
            url: Endpoint URL.
            payload: JSON payload.
            tool_name: Tên tool (chỉ để logging).

        Returns:
            Response dict nếu thành công, None nếu lỗi.
        """
        try:
            response = self._session.post(
                url,
                json=payload,
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            # Handle empty response body (202 Accepted, etc.)
            response_text = response.text.strip()
            if not response_text:
                logger.debug(
                    "MCP returned empty body: tool=%s, status=%d",
                    tool_name,
                    response.status_code,
                )
                return {"status": "accepted"}

            data: dict[str, Any] = response.json()
            return data

        except requests.Timeout:
            logger.warning(
                "MCP timeout sau %ds: tool=%s",
                _REQUEST_TIMEOUT,
                tool_name,
            )
        except requests.ConnectionError:
            logger.warning(
                "Không thể kết nối MCP server: %s. Kiểm tra server có đang chạy không.",
                self._base_url,
            )
        except requests.HTTPError as exc:
            logger.warning(
                "MCP HTTP error: tool=%s, status=%s",
                tool_name,
                exc.response.status_code if exc.response else "unknown",
            )
        except (requests.RequestException, json.JSONDecodeError) as exc:
            logger.warning(
                "MCP lỗi không xác định: tool=%s, error=%s",
                tool_name,
                exc,
            )

        return None

    @staticmethod
    def _extract_text_from_result(result: dict[str, Any]) -> str | None:
        """Trích xuất text content từ MCP response.

        MCP response có thể có nhiều format khác nhau.
        Hàm này thử các pattern phổ biến:
        1. result["content"][0]["text"]
        2. result["results"][0]["text"] hoặc ["document"]
        3. result["output"]

        Args:
            result: Response dict từ MCP server.

        Returns:
            Text content hoặc None nếu rỗng.
        """
        # Pattern 1: MCP standard content
        content = result.get("content", [])
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                text = first.get("text", "")
                if isinstance(text, str) and text.strip():
                    return text.strip()

        # Pattern 2: search results
        results = result.get("results", [])
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                for key in ("text", "document", "content"):
                    val = first.get(key, "")
                    if isinstance(val, str) and val.strip():
                        return val.strip()

        # Pattern 3: direct output
        output = result.get("output", "")
        if isinstance(output, str) and output.strip():
            return output.strip()

        return None

    @staticmethod
    def _extract_text_list_from_result(
        result: dict[str, Any],
        limit: int = 3,
    ) -> list[str]:
        """Trích xuất nhiều text snippets từ MCP response và loại trùng lặp."""
        snippets: list[str] = []

        content = result.get("content", [])
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text", "")
                if isinstance(text, str) and text.strip():
                    snippets.append(text.strip())

        results = result.get("results", [])
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                for key in ("text", "document", "content"):
                    value = item.get(key, "")
                    if isinstance(value, str) and value.strip():
                        snippets.append(value.strip())
                        break

        output = result.get("output", "")
        if isinstance(output, str) and output.strip():
            snippets.append(output.strip())

        unique = list(dict.fromkeys(snippets))
        return unique[: max(1, limit)]
