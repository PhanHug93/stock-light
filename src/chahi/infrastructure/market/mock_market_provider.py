"""Mock implementation của IMarketDataProvider cho mục đích testing/demo."""

from __future__ import annotations

import logging

from chahi.core.interfaces import IMarketDataProvider

logger = logging.getLogger(__name__)


class MockMarketDataProvider(IMarketDataProvider):
    """Giả lập dữ liệu thị trường cho các mã chứng khoán/vĩ mô chính."""

    def __init__(self) -> None:
        # Dữ liệu tĩnh cho demo
        self._prices: dict[str, float] = {
            "CL=F": 110.50,  # Crude Oil
            "DX-Y.NYB": 104.20,  # DXY
            "^TNX": 4.25,  # 10Y Yield
            "^VNI": 1250.0,  # VN-Index
            "GC=F": 2350.0,  # Gold
        }
        self._prev_prices: dict[str, float] = {
            "CL=F": 108.00,
            "DX-Y.NYB": 103.50,
            "^TNX": 4.15,
            "^VNI": 1260.0,
            "GC=F": 2360.0,
        }
        # Tương thích với get_macro keys
        self._macros: dict[str, float] = {
            "us10y_yield": 4.25,
            "dxy": 104.20,
            "oil_brent": 110.50,
        }

    def get_price(self, symbol: str) -> dict[str, float]:
        """Trả về giá hiện tại (giả lập)."""
        price = self._prices.get(symbol, 100.0)
        return {"close": price}

    def get_ohlcv(
        self,
        symbol: str,
        period: str = "5d",
    ) -> list[dict[str, float]]:
        """Trả về list các nến giả lập."""
        current = self._prices.get(symbol, 100.0)
        prev = self._prev_prices.get(symbol, 95.0)

        # Trả về 2 ngày cuối
        return [
            {
                "close": prev,
                "open": prev,
                "high": prev + 1,
                "low": prev - 1,
                "volume": 1000,
            },
            {
                "close": current,
                "open": current,
                "high": current + 1,
                "low": current - 1,
                "volume": 1100,
            },
        ]

    def get_macro(self, key: str) -> float:
        """Lấy chỉ số kinh tế vĩ mô giả lập."""
        return self._macros.get(key, 0.0)
