"""Scraper policy cho deep scraper và URL hygiene."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlparse

# Query params chỉ dùng tracking, cần loại khỏi cache key.
_TRACKING_QUERY_PARAMS: set[str] = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "_x_tr_sl",
}

# Từ khóa cốt lõi cho semantic budgeting.
_HOT_KEYWORDS: dict[str, int] = {
    "dxy": 20,
    "fed": 15,
    "fomc": 15,
    "cpi": 15,
    "inflation": 15,
    "oil": 15,
    "brent": 20,
    "wti": 20,
    "opec": 20,
    "yield": 12,
    "rate": 10,
    "earnings": 10,
    "bullish": 8,
    "bearish": 8,
}

# Domain policy cơ bản cho deep scrape.
_BLOCKED_PATHS: tuple[str, ...] = ("/video/", "/videos/", "/podcast/")
_PAYWALL_DOMAINS: set[str] = {"wsj.com", "bloomberg.com"}


def canonicalize_url(url: str) -> str:
    """Chuẩn hóa URL bằng blacklist tracking params.

    Chỉ loại các params marketing/tracking. Giữ lại params nghiệp vụ
    (ví dụ: ``id=123``) để tránh mất routing hợp lệ.
    """
    try:
        parsed = urlparse(url)
        query_params: list[tuple[str, str]] = []
        for k, v in parse_qsl(parsed.query, keep_blank_values=True):
            k_lower = k.lower()
            if k_lower in _TRACKING_QUERY_PARAMS or k_lower.startswith("utm_"):
                continue
            query_params.append((k, v))

        new_query = urlencode(query_params, doseq=True)
        return parsed._replace(query=new_query).geturl()
    except Exception:
        return url


def is_allowed_by_policy(url: str) -> bool:
    """Áp dụng domain policy cho deep scrape.

    Returns:
        True nếu an toàn để scrape, False nếu chặn.
    """
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path.lower()

        # Chặn video/podcast paths
        if any(
            path.startswith(blocked) or blocked in path for blocked in _BLOCKED_PATHS
        ):
            return False

        # Chặn known paywalls
        for paywall in _PAYWALL_DOMAINS:
            if host == paywall or host.endswith("." + paywall):
                return False

        return True
    except Exception:
        return False


def score_relevancy(title: str, summary: str) -> int:
    """Chấm điểm semantic relevance cho deep scrape budget.

    Returns:
        Score (càng cao càng ưu tiên deep scrape).
    """
    text = f"{title} {summary}".lower()
    score = 0

    for keyword, weight in _HOT_KEYWORDS.items():
        if keyword in text:
            score += weight

    if len(summary.strip()) < 80:
        score -= 5

    return score


def is_google_news_url(url: str) -> bool:
    """Kiểm tra URL có phải dạng Google News RSS redirect hay không."""
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    return host.endswith("news.google.com") and parsed.path.startswith("/rss/")
