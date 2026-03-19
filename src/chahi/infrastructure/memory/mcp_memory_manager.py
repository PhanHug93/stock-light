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
from typing import Any

import requests

from chahi.core.entities import MemorySettings
from chahi.core.interfaces import IMemoryManager

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

    def retrieve_last_context(self) -> str | None:
        """Truy xuất nhận định phiên gần nhất từ MCP server.

        Gọi tool ``search_memory`` với query tìm nhận định vĩ mô.
        Fallback: gọi ``auto_recall`` nếu search_memory thất bại.

        Returns:
            Nhận định cũ dạng text, hoặc None nếu không có/lỗi.
        """
        # ── Thử search_memory trước ──
        result = self._call_tool(
            tool_name="search_memory",
            arguments={
                "query": "nhận định phân tích vĩ mô phiên gần nhất ChaHi",
                "workspace_path": self._workspace_path,
                "n_results": 1,
            },
        )

        if result is not None:
            text = self._extract_text_from_result(result)
            if text:
                logger.info("  Retrieve từ search_memory: %d chars", len(text))
                return text

        # ── Fallback: auto_recall ──
        result = self._call_tool(
            tool_name="auto_recall",
            arguments={
                "user_message": "báo cáo phân tích vĩ mô hôm qua",
                "workspace_path": self._workspace_path,
                "n_results": 1,
            },
        )

        if result is not None:
            text = self._extract_text_from_result(result)
            if text:
                logger.info("  Retrieve từ auto_recall: %d chars", len(text))
                return text

        logger.info("  Không tìm thấy nhận định cũ trong MCP.")
        return None

    def save_context(self, context_data: str) -> None:
        """Lưu nhận định mới vào MCP server.

        Gọi tool ``store_working_context`` để lưu vào L1 (per-workspace).

        Args:
            context_data: Nội dung nhận định cần lưu.
        """
        result = self._call_tool(
            tool_name="store_working_context",
            arguments={
                "text_data": context_data,
                "metadata_source": "chahi_macro_report",
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
            logger.warning("SSE stream kết thúc mà không nhận được result cho tool=%s", tool_name)
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
                "Không thể kết nối MCP server: %s. "
                "Kiểm tra server có đang chạy không.",
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
