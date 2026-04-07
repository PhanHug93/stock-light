"""Divergence Engine — Trích xuất sự lệch pha giữa các biến số vĩ mô/tài sản.

Phân kỳ xảy ra khi hai chỉ báo logic đồng thời di chuyển
ngược chiều nhau (ví dụ Tin tốt nhưng Giá giảm).
Đây là alpha signal mạnh nhất trong phân tích macro.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DivergenceFlag:
    """Một cờ phân kỳ cụ thể."""

    code: str
    description: str
    severity: str = "medium"  # low / medium / high


class DivergenceEngine:
    """Truy tìm phân kỳ để tạo cờ cảnh báo (Flags).

    Tất cả logic là deterministic — không sử dụng LLM.
    """

    def evaluate(
        self,
        features: dict[str, Any],
    ) -> list[DivergenceFlag]:
        """Chạy toàn bộ divergence checks.

        Args:
            features: Dict chứa features từ FeatureEngine.

        Returns:
            Danh sách DivergenceFlag nếu phát hiện phân kỳ.
        """
        flags: list[DivergenceFlag] = []

        oil_up = features.get("oil_return_1d", 0.0) > 0
        oil_stocks_return = features.get("vn_oil_stocks_return_1d", 0.0)
        news_bullish = features.get("news_positive_count", 0) > features.get(
            "news_negative_count", 0
        )
        market_down = features.get("vnindex_return_1d", 0.0) < 0
        dxy_up = features.get("dxy_change", 0.0) > 0
        vnindex_up = features.get("vnindex_return_1d", 0.0) > 0

        # Divergence #1 — Oil vs Oil Stocks
        if oil_up and oil_stocks_return < 0:
            flags.append(
                DivergenceFlag(
                    code="OIL_STOCK_DIVERGENCE",
                    description=(
                        "Giá dầu tăng nhưng cổ phiếu dầu khí VN giảm — "
                        "market chưa pricing kịp hoặc false breakout oil"
                    ),
                    severity="high",
                )
            )

        # Divergence #2 — News vs Price
        if news_bullish and market_down:
            flags.append(
                DivergenceFlag(
                    code="NEWS_PRICE_DIVERGENCE",
                    description=(
                        "Tin tốt ngập tràn nhưng thị trường bị bán tháo — "
                        "smart money đang phân phối?"
                    ),
                    severity="high",
                )
            )

        # Divergence #3 — DXY vs VNIndex
        if dxy_up and vnindex_up:
            flags.append(
                DivergenceFlag(
                    code="LIQUIDITY_ANOMALY",
                    description=(
                        "DXY tăng nhưng VNIndex vẫn tăng — "
                        "bất thường thanh khoản, cần theo dõi FII flow"
                    ),
                    severity="medium",
                )
            )

        return flags
