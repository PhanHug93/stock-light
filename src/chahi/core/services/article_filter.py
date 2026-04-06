"""Article Filter Service — lọc keyword và khử trùng lặp trước MAP.

Mục tiêu:
1. Giảm noise trước khi đưa dữ liệu vào LLM.
2. Hạn chế trùng lặp giữa nhiều RSS feeds cùng đưa một sự kiện.
3. Trích xuất hot keywords phục vụ semantic retrieval từ Memory.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from chahi.core.entities import AnalysisContext, Article, SourceCategory

_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹ]+")

_DEFAULT_KEYWORDS: dict[SourceCategory, set[str]] = {
    SourceCategory.OIL_MACRO: {
        "fed",
        "lãi suất",
        "inflation",
        "cpi",
        "pmi",
        "opec",
        "opec+",
        "brent",
        "wti",
        "oil",
        "crude",
        "dxy",
        "usd",
        "usd/vnd",
        "gdp",
        "jobs",
        "payroll",
    },
    SourceCategory.GOLD: {
        "gold",
        "vàng",
        "xau",
        "bullion",
        "ounce",
        "safe haven",
        "real yield",
        "etf",
        "kitco",
        "dxy",
    },
    SourceCategory.CRYPTO: {
        "crypto",
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "stablecoin",
        "defi",
        "etf",
        "onchain",
        "halving",
        "altcoin",
        "token",
    },
}

_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "will",
    "your",
    "của",
    "cho",
    "với",
    "đang",
    "những",
    "trong",
    "được",
    "một",
    "các",
    "khi",
    "này",
}


@dataclass(frozen=True, slots=True)
class FilterStats:
    """Thống kê filter theo category."""

    input_count: int
    keyword_kept: int
    dedup_removed: int
    output_count: int


class ArticleFilterService:
    """Lọc keyword + dedup dựa trên Jaccard Similarity."""

    def __init__(
        self,
        similarity_threshold: float = 0.70,
        keywords: dict[SourceCategory, set[str]] | None = None,
    ) -> None:
        if not (0.0 < similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold phải nằm trong (0.0, 1.0].")
        raw_keywords = keywords or _DEFAULT_KEYWORDS
        self._keywords = {
            category: {keyword.strip().lower() for keyword in values if keyword.strip()}
            for category, values in raw_keywords.items()
        }
        self._similarity_threshold = similarity_threshold

    def filter_context(self, context: AnalysisContext) -> AnalysisContext:
        """Lọc toàn bộ AnalysisContext theo 3 category."""
        oil_filtered, oil_stats = self.filter_articles(
            SourceCategory.OIL_MACRO,
            list(context.oil_news),
        )
        gold_filtered, gold_stats = self.filter_articles(
            SourceCategory.GOLD,
            list(context.gold_news),
        )
        crypto_filtered, crypto_stats = self.filter_articles(
            SourceCategory.CRYPTO,
            list(context.crypto_news),
        )

        self._log_stats("OIL_MACRO", oil_stats)
        self._log_stats("GOLD", gold_stats)
        self._log_stats("CRYPTO", crypto_stats)

        return AnalysisContext(
            date=context.date,
            oil_news=oil_filtered,
            gold_news=gold_filtered,
            crypto_news=crypto_filtered,
            previous_context=context.previous_context,
        )

    def filter_articles(
        self,
        category: SourceCategory,
        articles: list[Article],
    ) -> tuple[list[Article], FilterStats]:
        """Lọc keyword trước, sau đó dedup theo Jaccard Similarity."""
        if not articles:
            stats = FilterStats(
                input_count=0,
                keyword_kept=0,
                dedup_removed=0,
                output_count=0,
            )
            return [], stats

        keyword_candidates: list[tuple[Article, set[str], str]] = []
        for article in articles:
            normalized_text = self._normalize(f"{article.title} {article.summary}")
            token_set = self._tokenize(normalized_text)

            if self._matches_keywords(category, token_set, normalized_text):
                keyword_candidates.append((article, token_set, normalized_text))

        keyword_kept = len(keyword_candidates)

        deduped: list[tuple[Article, set[str], str]] = []
        dedup_removed = 0
        for article, token_set, normalized_text in keyword_candidates:
            duplicate_idx = self._find_duplicate_index(token_set, deduped)
            if duplicate_idx is None:
                deduped.append((article, token_set, normalized_text))
                continue

            dedup_removed += 1
            existing_article, _existing_tokens, _existing_text = deduped[duplicate_idx]
            if self._article_score(article) > self._article_score(existing_article):
                deduped[duplicate_idx] = (article, token_set, normalized_text)

        filtered = [article for article, _tokens, _text in deduped]
        stats = FilterStats(
            input_count=len(articles),
            keyword_kept=keyword_kept,
            dedup_removed=dedup_removed,
            output_count=len(filtered),
        )
        return filtered, stats

    def extract_hot_keywords(
        self,
        context: AnalysisContext,
        max_keywords: int = 8,
    ) -> list[str]:
        """Trích hot keywords từ dữ liệu đã lọc để semantic retrieval."""
        if max_keywords <= 0:
            return []

        frequency: Counter[str] = Counter()
        for category, articles in self._iter_categories(context):
            keywords = self._keywords.get(category, set())
            for article in articles:
                normalized = self._normalize(f"{article.title} {article.summary}")
                tokens = self._tokenize(normalized)
                for keyword in keywords:
                    if " " in keyword:
                        if keyword in normalized:
                            frequency[keyword] += 1
                    elif keyword in tokens:
                        frequency[keyword] += 1

        if frequency:
            return [keyword for keyword, _count in frequency.most_common(max_keywords)]

        fallback: Counter[str] = Counter()
        for _category, articles in self._iter_categories(context):
            for article in articles:
                for token in self._tokenize(self._normalize(article.title)):
                    if len(token) < 4 or token in _STOP_WORDS:
                        continue
                    fallback[token] += 1
        return [keyword for keyword, _count in fallback.most_common(max_keywords)]

    def _matches_keywords(
        self,
        category: SourceCategory,
        token_set: set[str],
        normalized_text: str,
    ) -> bool:
        keywords = self._keywords.get(category, set())
        if not keywords:
            return True

        for keyword in keywords:
            if " " in keyword:
                if keyword in normalized_text:
                    return True
            elif keyword in token_set:
                return True
        return False

    def _find_duplicate_index(
        self,
        token_set: set[str],
        deduped: list[tuple[Article, set[str], str]],
    ) -> int | None:
        for idx, (_existing_article, existing_tokens, _existing_text) in enumerate(
            deduped
        ):
            similarity = self._jaccard(token_set, existing_tokens)
            if similarity >= self._similarity_threshold:
                return idx
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(_TOKEN_RE.findall(text))

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        intersection = len(a.intersection(b))
        union = len(a.union(b))
        if union == 0:
            return 0.0
        return intersection / union

    @staticmethod
    def _article_score(article: Article) -> int:
        return len(article.title) + len(article.summary)

    @staticmethod
    def _iter_categories(
        context: AnalysisContext,
    ) -> list[tuple[SourceCategory, list[Article]]]:
        return [
            (SourceCategory.OIL_MACRO, list(context.oil_news)),
            (SourceCategory.GOLD, list(context.gold_news)),
            (SourceCategory.CRYPTO, list(context.crypto_news)),
        ]

    @staticmethod
    def _log_stats(category_label: str, stats: FilterStats) -> None:
        from logging import getLogger

        logger = getLogger(__name__)
        logger.info(
            ("Filter [%s]: input=%d, keyword_kept=%d, dedup_removed=%d, output=%d"),
            category_label,
            stats.input_count,
            stats.keyword_kept,
            stats.dedup_removed,
            stats.output_count,
        )
