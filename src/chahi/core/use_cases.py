"""Use Cases — Application logic ghép nối tất cả components.

Module này chứa các use cases (application services) là nơi
orchestrate workflow chính của ChaHi. Mỗi use case nhận
dependencies qua constructor (DI) và expose một method ``execute()``.

Quy tắc:
    - Use case chỉ phụ thuộc vào core/ interfaces và entities.
    - Không import trực tiếp infrastructure modules.
    - Prompts nằm ở ``core/prompts.py`` (SRP).
    - Report parsing nằm ở ``core/services/report_parser.py`` (SRP).

Pipeline Phase 8 (Map-Reduce + Feedback Loop):
    1. **Retrieve**: Lấy nhận định cũ từ Memory.
    2. **Fetch**: Cào tin tức song song từ RSS.
    3. **Contextualize**: Trích hot keywords + semantic retrieval từ Memory.
    4. **MAP**: Gọi LLM phân tích từng nhóm (Dầu, Vàng, Crypto).
    5. **REDUCE**: Gọi LLM tổng hợp 3 bản tóm tắt + Memory → Báo cáo cuối.
    6. **Store**: Lưu phần Tổng kết vào Memory.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import TYPE_CHECKING, Any

from chahi.core.entities import AnalysisContext, SourceCategory
from chahi.core.interfaces import (
    IConfigReader,
    ILLMClient,
    IMemoryManager,
    INewsFetcher,
)
from chahi.core.prompts import CATEGORY_NAMES, MAP_PROMPT, REDUCE_PROMPT
from chahi.core.services.article_filter import ArticleFilterService
from chahi.core.services.divergence_engine import DivergenceEngine, DivergenceFlag
from chahi.core.services.feature_engine import FeatureEngine
from chahi.core.services.report_parser import extract_summary
from chahi.core.services.rule_engine import RuleEngine, RuleSignals

if TYPE_CHECKING:
    from chahi.core.entities import Article, SourceConfig
    from chahi.core.interfaces import (
        IConfigReader,
        ILLMClient,
        IMarketDataProvider,
        IMemoryManager,
        INewsFetcher,
    )

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════
# Prompts & parsing logic đã được tách ra:
#   - core/prompts.py          → MAP_PROMPT, REDUCE_PROMPT, CATEGORY_NAMES
#   - core/services/report_parser.py → extract_summary()
# ═════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════
# Use Case
# ═════════════════════════════════════════════════════════════


class GenerateMacroReportUseCase:
    """Use case: Tạo báo cáo phân tích vĩ mô (Map-Reduce + Memory).

    Pipeline Phase 8:
    1. **Retrieve**: Lấy nhận định ngày trước từ ``IMemoryManager``.
    2. Đọc cấu hình nguồn tin từ ``IConfigReader``.
    3. Fetch tin tức song song từ ``INewsFetcher``.
    4. **Contextualize**: Trích hot keywords và truy xuất bài học liên quan.
    5. **MAP**: Gọi LLM phân tích từng nhóm (3 calls).
    6. **REDUCE**: Gọi LLM tổng hợp → Báo cáo cuối + Vietnam focus.
    7. **Store**: Lưu phần Tổng kết vào Memory.

    Args:
        config_reader: Component đọc cấu hình (DI).
        news_fetcher: Component cào tin tức (DI).
        llm_client: Component giao tiếp LLM (DI).
        memory_manager: Component quản lý bộ nhớ dài hạn (DI, optional).
    """

    def __init__(
        self,
        config_reader: IConfigReader,
        news_fetcher: INewsFetcher,
        llm_client: ILLMClient,
        market_data_provider: IMarketDataProvider | None = None,
        memory_manager: IMemoryManager | None = None,
        article_filter: ArticleFilterService | None = None,
        feature_engine: FeatureEngine | None = None,
        rule_engine: RuleEngine | None = None,
        divergence_engine: DivergenceEngine | None = None,
    ) -> None:
        self._config_reader = config_reader
        self._news_fetcher = news_fetcher
        self._llm_client = llm_client
        self._market_data = market_data_provider
        self._memory = memory_manager
        self._article_filter = article_filter or ArticleFilterService()
        self._feature_engine = feature_engine or FeatureEngine()
        self._rule_engine = rule_engine or RuleEngine()
        self._divergence_engine = divergence_engine or DivergenceEngine()

    def execute(self) -> str:
        """Chạy pipeline Map-Reduce với Feedback Loop.

        Returns:
            Nội dung báo cáo Markdown từ LLM.

        Raises:
            ConnectionError: Khi không kết nối được RSS hoặc LLM.
            RuntimeError: Khi LLM trả về response không hợp lệ.
            ValueError: Khi config không hợp lệ.
        """
        # ── Bước 1: RETRIEVE — lấy nhận định cũ từ Memory ──
        previous_context: str | None = None
        if self._memory is not None:
            logger.info("Bước 1/7: Truy xuất nhận định cũ từ Memory...")
            try:
                previous_context = self._memory.retrieve_last_context()
                if previous_context:
                    logger.info("  → Có nhận định cũ: %d chars", len(previous_context))
                else:
                    logger.info("  → Chưa có nhận định cũ (lần đầu chạy).")
            except Exception as exc:
                logger.warning("  → Lỗi đọc Memory, bỏ qua: %s", exc)
        else:
            logger.info("Bước 1/7: Memory Manager chưa được cấu hình, bỏ qua.")

        # ── Bước 2: Đọc config nguồn tin ──
        logger.info("Bước 2/7: Đọc cấu hình nguồn tin...")
        sources = self._config_reader.get_sources()

        # ── Bước 3: Fetch tin tức ──
        logger.info("Bước 3/7: Cào tin tức từ %d categories...", len(sources))
        context = self._fetch_all_news(sources, previous_context)
        if context.is_empty:
            logger.warning("Không có bài viết nào được fetch. Dừng pipeline.")
            return "⚠️ Không có tin tức nào để phân tích."

        logger.info(
            "Tổng cộng %d bài viết: Oil&Macro=%d, Gold=%d, Crypto=%d",
            context.total_articles,
            len(context.oil_news),
            len(context.gold_news),
            len(context.crypto_news),
        )

        # ── Bước 4: Contextualize — hot keywords + semantic memory ──
        hot_keywords = self._article_filter.extract_hot_keywords(
            context, max_keywords=10
        )
        if hot_keywords:
            logger.info("Hot keywords: %s", ", ".join(hot_keywords))
        else:
            logger.info("Hot keywords: không trích được keyword nổi bật.")

        semantic_context: str | None = None
        if self._memory is not None and hot_keywords:
            logger.info("Bước 4/7: Truy xuất bài học liên quan (semantic memory)...")
            try:
                semantic_raw = self._memory.retrieve_related_context(
                    hot_keywords=hot_keywords,
                    max_results=5,
                )
                if isinstance(semantic_raw, str) and semantic_raw.strip():
                    semantic_context = semantic_raw.strip()
                    logger.info(
                        "  → Tìm thấy bài học liên quan: %d chars",
                        len(semantic_context),
                    )
                else:
                    logger.info("  → Không có bài học liên quan từ semantic retrieval.")
            except Exception as exc:
                logger.warning("  → Lỗi semantic retrieval, bỏ qua: %s", exc)
        else:
            logger.info(
                "Bước 4/7: Bỏ qua semantic retrieval (không có memory/keywords)."
            )

        combined_context = self._merge_contexts(previous_context, semantic_context)

        # ── Bước 5: MAP — Phân tích cục bộ từng nhóm ──
        logger.info("Bước 5/7: MAP — Phân tích từng nhóm...")
        category_summaries = self._map_analyze_all(context)
        if not any(category_summaries.values()):
            logger.error("MAP thất bại hoàn toàn. Không có tóm tắt nào.")
            return "⚠️ Không thể phân tích tin tức (MAP failed)."

        # ── Bước 6: QUANT — Định lường & Đối chiếu (Feature, Rule, Divergence) ──
        logger.info("Bước 6/7: QUANT — Chạy Rule & Divergence Engine...")
        rule_signals: RuleSignals | None = None
        divergence_flags: list[DivergenceFlag] = []

        if self._market_data:
            try:
                # 1. Lấy dữ liệu thị trường thực tế
                raw_market = self._fetch_market_data()
                # 2. Xử lý Features (Market + NLP)
                # Note: sentiment có thể lấy từ summaries nếu LLM hỗ trợ return JSON,
                # ở đây ta tạm thời dùng counts đơn giản từ filter hoặc mock.
                nlp_output = {
                    "news_positive_count": 0,
                    "news_negative_count": 0,
                }  # Placeholder
                features = self._feature_engine.compute(raw_market, nlp_output)
                # 3. Rule Engine (Deterministic)
                rule_signals = self._rule_engine.evaluate(features)
                # 4. Divergence Engine (Anomaly detection)
                divergence_flags = self._divergence_engine.evaluate(features)
                logger.info("  → Quant analysis hoàn tất.")
            except Exception as exc:
                logger.warning("  → Quant Layer lỗi, tiếp tục dùng AI thuần: %s", exc)

        # ── Bước 7: REDUCE — Tổng hợp & Bản địa hóa ──
        logger.info("Bước 7/7: REDUCE — Tổng hợp & Bản địa hóa VN...")
        reduce_input = self._format_reduce_input(
            category_summaries,
            previous_context=combined_context,
            hot_keywords=hot_keywords,
            rule_signals=rule_signals,
            divergence_flags=divergence_flags,
        )
        report = self._llm_client.analyze(
            system_prompt=REDUCE_PROMPT,
            user_content=reduce_input,
        )
        logger.info("✓ Báo cáo đã được tạo: %d chars", len(report))

        # ── Bước 7: STORE — Lưu nhận định mới vào Memory ──
        if self._memory is not None:
            logger.info("Bước 7/7: Lưu nhận định mới vào Memory...")
            try:
                summary = extract_summary(report)
                self._memory.save_context(summary)
                logger.info("  → Đã lưu %d chars vào Memory.", len(summary))
            except Exception as exc:
                logger.warning("  → Lỗi lưu Memory, bỏ qua: %s", exc)
        else:
            logger.info("Bước 7/7: Memory Manager chưa được cấu hình, bỏ qua.")

        return report

    @staticmethod
    def _merge_contexts(
        previous_context: str | None,
        semantic_context: str | None,
    ) -> str | None:
        """Ghép context gần nhất + context semantic, tránh trùng lặp."""
        previous = (previous_context or "").strip()
        semantic = (semantic_context or "").strip()

        if previous and semantic:
            if semantic in previous:
                return previous
            if previous in semantic:
                return semantic
            return (
                "### [NHÌN LẠI QUÁ KHỨ] — Phiên gần nhất\n"
                f"{previous}\n\n"
                "### [BÀI HỌC LIÊN QUAN] — Truy xuất theo hot keywords\n"
                f"{semantic}"
            )
        if previous:
            return previous
        if semantic:
            return semantic
        return None

    # ── MAP: Phân tích từng nhóm ─────────────────────────────

    _MAX_MAP_WORKERS: int = 3  # 3 nhóm tài sản song song

    def _analyze_category(self, category_name: str, articles: list[Article]) -> str:
        """MAP: Phân tích cục bộ 1 nhóm tài sản.

        Gửi danh sách articles cùng MAP_PROMPT tới LLM
        và nhận bản tóm tắt 3-5 bullet points.

        Args:
            category_name: Tên nhóm (vd: "Dầu & Kinh tế Vĩ mô").
            articles: Danh sách bài viết của nhóm.

        Returns:
            Bản tóm tắt từ LLM, hoặc chuỗi rỗng nếu lỗi.
        """
        if not articles:
            return ""

        # Format articles → semantic tags
        lines: list[str] = [
            "<filter_instruction>",
            "- Không loại bài khỏi input; chỉ giảm trọng số tin nhiễu trong lập luận.",
            "- Nếu nhiều bài cùng một sự kiện, hợp nhất ý và giữ nguồn tin cậy hơn.",
            "- Giữ nguyên số liệu định lượng quan trọng từ các bài trùng.",
            "</filter_instruction>",
            f'<category name="{category_name}">',
        ]
        for i, article in enumerate(articles, 1):
            pub_date = article.published_date.strftime("%d/%m/%Y %H:%M")
            lines.append(f'  <article id="{i}">')
            lines.append(f"    <title>{article.title}</title>")
            lines.append(f"    <source>{article.source_name}</source>")
            lines.append(f"    <published_at>{pub_date}</published_at>")
            lines.append(f"    <summary>{article.summary}</summary>")
            lines.append("  </article>")
        lines.append("</category>")

        user_content = "\n".join(lines)
        system_prompt = MAP_PROMPT.format(category_name=category_name)

        try:
            result = self._llm_client.analyze(
                system_prompt=system_prompt,
                user_content=user_content,
            )
            logger.info("  MAP [%s] → %d chars tóm tắt", category_name, len(result))
            return result
        except Exception as exc:
            logger.warning("  MAP [%s] thất bại: %s", category_name, exc)
            return ""

    def _map_analyze_all(self, context: AnalysisContext) -> dict[SourceCategory, str]:
        """Chạy MAP cho tất cả danh mục tin tức.

        Tự động chọn chế độ:
        - Cloud LLM (Gemini): song song qua ThreadPoolExecutor.
        - Local LLM (LM Studio): tuần tự để tránh nghẽn queue.

        Args:
            context: AnalysisContext chứa tin tức phân nhóm theo category.

        Returns:
            Dict mapping SourceCategory → bản tóm tắt LLM.
        """
        tasks: list[tuple[SourceCategory, list[Article]]] = [
            (cat, list(articles))
            for cat, articles in context.news_by_category.items()
            if articles  # Chỉ phân tích nhóm có tin
        ]

        results: dict[SourceCategory, str] = {}

        if self._llm_client.supports_concurrency:
            # ── Cloud LLM: song song (auto-scaling) ──
            logger.info("  MAP mode: PARALLEL (cloud LLM)")
            with ThreadPoolExecutor(max_workers=self._MAX_MAP_WORKERS) as pool:
                future_map = {
                    pool.submit(self._analyze_category, name, articles): name
                    for name, articles in tasks
                }

                for future in as_completed(future_map):
                    name = future_map[future]
                    try:
                        results[name] = future.result()
                    except Exception as exc:
                        logger.warning("MAP [%s] exception: %s", name, exc)
                        results[name] = ""
        else:
            # ── Local LLM: tuần tự (queue-based, tránh timeout) ──
            logger.info("  MAP mode: SEQUENTIAL (local LLM)")
            for name, articles in tasks:
                try:
                    results[name] = self._analyze_category(name, articles)
                except Exception as exc:
                    logger.warning("MAP [%s] exception: %s", name, exc)
                    results[name] = ""

        return results

    # ── REDUCE: Format input ─────────────────────────────────

    def _format_reduce_input(
        self,
        summaries: dict[SourceCategory, str],
        previous_context: str | None = None,
        hot_keywords: list[str] | None = None,
        rule_signals: RuleSignals | None = None,
        divergence_flags: list[DivergenceFlag] | None = None,
    ) -> str:
        """Format dữ liệu cho REDUCE prompt."""
        parts = [f"<analysis_date>{date.today().strftime('%d/%m/%Y')}</analysis_date>"]

        if hot_keywords:
            parts.append(f"<hot_keywords>\n{', '.join(hot_keywords)}\n</hot_keywords>")

        if previous_context:
            parts.append(f"<previous_lessons>\n{previous_context}\n</previous_lessons>")

        if rule_signals:
            signals_str = "\n".join(
                f"- {k.replace('_', ' ').title()}: {v}"
                for k, v in rule_signals.to_dict().items()
            )
            parts.append(f"<rule_signals>\n{signals_str}\n</rule_signals>")

        if divergence_flags:
            flags_str = "\n".join(
                f"- [{f.severity.upper()}] {f.description}" for f in divergence_flags
            )
            parts.append(f"<divergence_alerts>\n{flags_str}\n</divergence_alerts>")

        parts.append("<map_summaries>")
        for cat, summary in summaries.items():
            cat_name = CATEGORY_NAMES.get(cat, cat.name)
            parts.append(
                f'  <category name="{cat_name}">\n{summary.strip()}\n  </category>'
            )
        parts.append("</map_summaries>")

        return "\n\n".join(parts)

    def _fetch_market_data(self) -> dict[str, Any]:
        """Lấy dữ liệu thô từ provider cho các mã quan trọng."""
        if not self._market_data:
            return {}

        data: dict[str, Any] = {}
        symbols = {
            "oil_close": "CL=F",
            "dxy_close": "DX-Y.NYB",
            "yield_10y": "^TNX",
            "vnindex_close": "^VNI",
        }

        for key, sym in symbols.items():
            try:
                price = self._market_data.get_price(sym)
                data[key] = price.get("close")
                # Lấy thêm giá hôm qua để tính return
                ohlcv = self._market_data.get_ohlcv(sym, period="2d")
                if len(ohlcv) >= 2:
                    data[key.replace("_close", "_prev_close")] = ohlcv[0].get("close")
            except Exception as exc:
                logger.debug("Không thể lấy market data cho %s (%s): %s", key, sym, exc)

        return data

    # ── Fetch tin tức ─────────────────────────────────────────

    def _fetch_all_news(
        self,
        sources: dict[SourceCategory, list[SourceConfig]],
        previous_context: str | None = None,
    ) -> AnalysisContext:
        """Fetch tin tức từ tất cả sources song song.

        Args:
            sources: Dict category → list[SourceConfig] từ config.
            previous_context: Nhận định cũ từ Memory (nếu có).

        Returns:
            AnalysisContext chứa tin tức + previous_context.
        """
        category_map: dict[SourceCategory, list[Article]] = {cat: [] for cat in sources}

        # Chuẩn bị danh sách tasks
        tasks: list[tuple[SourceCategory, SourceConfig]] = []
        for category, source_list in sources.items():
            for source in source_list:
                tasks.append((category, source))

        # Ưu tiên fetch_many (async/batch) nếu fetcher hỗ trợ.
        if self._news_fetcher.supports_batch:
            try:
                batch_result = self._news_fetcher.fetch_many(tasks=tasks, limit=10)
                for cat, articles in batch_result.items():
                    category_map.setdefault(cat, []).extend(articles)
                return AnalysisContext(
                    date=date.today(),
                    news_by_category=category_map,
                    previous_context=previous_context,
                )
            except Exception as exc:
                logger.warning("fetch_many thất bại, fallback tuần tự: %s", exc)

        for category, source in tasks:
            target = category_map.setdefault(category, [])
            try:
                articles = self._news_fetcher.fetch_news(
                    url=source.url,
                    limit=10,
                )
                target.extend(articles)
                logger.info(
                    "  [%s] %s → %d bài",
                    category,
                    source.name,
                    len(articles),
                )
            except ConnectionError as exc:
                logger.warning(
                    "  [%s] %s → Lỗi kết nối: %s",
                    category,
                    source.name,
                    exc,
                )
            except Exception as exc:
                logger.warning(
                    "  [%s] %s → Lỗi không xác định: %s",
                    category,
                    source.name,
                    exc,
                )

        return AnalysisContext(
            date=date.today(),
            news_by_category=category_map,
            previous_context=previous_context,
        )


# _extract_summary đã chuyển sang core/services/report_parser.py
