"""Feature Engine — Xử lý dữ liệu thô thành features cho Rule Engine.

Tầng này nhận Market Data (giá, volume) và NLP output (sentiment counts)
rồi tính toán các features chuẩn hóa phục vụ cho RuleEngine và DivergenceEngine.

Tất cả logic là Python thuần — không có LLM, đảm bảo deterministic.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Tính toán features từ market data và NLP output.

    Đầu vào: Raw market data từ IMarketDataProvider + sentiment counts.
    Đầu ra: Flat dict[str, Any] chứa features chuẩn hóa.
    """

    def compute(
        self,
        market_data: dict[str, Any] | None = None,
        nlp_output: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Tính toán feature vector.

        Args:
            market_data: Raw data từ IMarketDataProvider.
                Expected keys: "oil_close", "oil_prev_close",
                "dxy_close", "dxy_prev_close", "yield_10y",
                "vnindex_close", "vnindex_prev_close",
                "vnindex_volume", "vnindex_volume_avg_20d",
                "vn_oil_stocks_close", "vn_oil_stocks_prev_close".
            nlp_output: Sentiment counts từ NLP layer.
                Expected keys: "news_positive_count", "news_negative_count".

        Returns:
            Flat dict chứa computed features.
        """
        market = market_data or {}
        nlp = nlp_output or {}
        features: dict[str, Any] = {}

        # ── Market-derived features ──
        features["oil_return_1d"] = self._safe_return(
            market.get("oil_close"), market.get("oil_prev_close")
        )
        features["dxy_change"] = self._safe_return(
            market.get("dxy_close"), market.get("dxy_prev_close")
        )
        features["yield_10y"] = market.get("yield_10y", 0.0)
        features["vnindex_return_1d"] = self._safe_return(
            market.get("vnindex_close"), market.get("vnindex_prev_close")
        )
        features["vnindex_volume"] = market.get("vnindex_volume", 0)
        features["vnindex_volume_avg_20d"] = market.get(
            "vnindex_volume_avg_20d", float("inf")
        )
        features["vn_oil_stocks_return_1d"] = self._safe_return(
            market.get("vn_oil_stocks_close"),
            market.get("vn_oil_stocks_prev_close"),
        )

        # ── NLP-derived features ──
        features["news_positive_count"] = nlp.get("news_positive_count", 0)
        features["news_negative_count"] = nlp.get("news_negative_count", 0)

        return features

    @staticmethod
    def _safe_return(
        current: float | None,
        previous: float | None,
    ) -> float:
        """Tính return an toàn, trả 0.0 nếu thiếu dữ liệu."""
        if current is None or previous is None or previous == 0:
            return 0.0
        return (current - previous) / previous
