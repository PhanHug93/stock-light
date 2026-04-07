"""Rule Engine — Tổng hợp tín hiệu từ Market Data & NLP Features.

Chạy các Rules cố định (Hard-coded) để tránh bị ảo giác từ LLM.
Đầu vào: Các features (Tỉ giá, News count, Giá thị trường).
Đầu ra: Tập hợp tín hiệu (RuleSignals).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RuleSignals:
    """Tập hợp các tín hiệu sau khi chạy Rule Engine."""

    oil: str = "neutral"
    vn_oil_gas: str = "neutral"
    vnindex: str = "neutral"
    bank: str = "neutral"
    equity_global: str = "neutral"
    trend_strength: str = "neutral"
    sentiment: str = "neutral"

    def to_dict(self) -> dict[str, str]:
        """Serialize signals thành dict để inject vào REDUCE prompt."""
        return {
            "oil": self.oil,
            "vn_oil_gas": self.vn_oil_gas,
            "vnindex": self.vnindex,
            "bank": self.bank,
            "equity_global": self.equity_global,
            "trend_strength": self.trend_strength,
            "sentiment": self.sentiment,
        }


class RuleEngine:
    """Đánh giá các quy tắc tĩnh dựa trên dữ liệu Features.

    Mỗi Rule là một hàm thuần túy: input → output, không có
    side-effect, luôn cho kết quả giống nhau (Deterministic).
    """

    def evaluate(self, features: dict[str, Any]) -> RuleSignals:
        """Chạy toàn bộ rule set và trả về signals.

        Args:
            features: Dict chứa các feature đã được FeatureEngine tính toán.

        Returns:
            RuleSignals chứa tín hiệu cho từng lớp tài sản.
        """
        oil = "neutral"
        vn_oil_gas = "neutral"
        vnindex = "neutral"
        bank = "neutral"
        equity_global = "neutral"
        trend_strength = "neutral"
        sentiment = "neutral"

        # Rule #1 — Oil Shock (Geopolitics proxy)
        oil_return_1d = features.get("oil_return_1d", 0.0)
        if oil_return_1d > 0.03:  # >3%
            oil = "bullish"
            vn_oil_gas = "bullish"
        elif oil_return_1d < -0.03:
            oil = "bearish"

        # Rule #2 — DXY Pressure (Capital flow)
        dxy_change = features.get("dxy_change", 0.0)
        if dxy_change > 0.005:  # >0.5%
            vnindex = "bearish"
            bank = "bearish"
        elif dxy_change < -0.005:
            vnindex = "bullish"

        # Rule #3 — Yield Shock (Risk-off trigger)
        us10y = features.get("yield_10y", 0.0)
        if us10y > 4.5:
            equity_global = "bearish"

        # Rule #4 — Volume Confirmation (VNIndex)
        vnindex_up = features.get("vnindex_return_1d", 0.0) > 0
        volume_above_avg = features.get("vnindex_volume", 0) > features.get(
            "vnindex_volume_avg_20d", float("inf")
        )
        trend_strength = "confirmed" if vnindex_up and volume_above_avg else "weak"

        # Rule #5 — News Pressure (NLP output)
        news_negative_count = features.get("news_negative_count", 0)
        if news_negative_count > 3:
            sentiment = "bearish"
        elif features.get("news_positive_count", 0) > 3:
            sentiment = "bullish"

        return RuleSignals(
            oil=oil,
            vn_oil_gas=vn_oil_gas,
            vnindex=vnindex,
            bank=bank,
            equity_global=equity_global,
            trend_strength=trend_strength,
            sentiment=sentiment,
        )
