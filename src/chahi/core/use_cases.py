"""Use Cases — Application logic ghép nối tất cả components.

Module này chứa các use cases (application services) là nơi
orchestrate workflow chính của ChaHi. Mỗi use case nhận
dependencies qua constructor (DI) và expose một method ``execute()``.

Quy tắc:
    - Use case chỉ phụ thuộc vào core/ interfaces và entities.
    - Không import trực tiếp infrastructure modules.
    - Prompts được định nghĩa tại đây vì chúng là application logic.

Pipeline Phase 8 (Map-Reduce + Feedback Loop):
    1. **Retrieve**: Lấy nhận định cũ từ Memory.
    2. **Fetch**: Cào tin tức song song từ RSS.
    3. **Filter**: Lọc từ khóa + khử trùng lặp bằng Jaccard Similarity.
    4. **Semantic Retrieve**: Trích hot keywords để lấy bài học liên quan.
    5. **MAP**: Gọi LLM phân tích từng nhóm (Dầu, Vàng, Crypto).
    6. **REDUCE**: Gọi LLM tổng hợp 3 bản tóm tắt + Memory → Báo cáo cuối.
    7. **Store**: Lưu phần Tổng kết vào Memory.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import TYPE_CHECKING

from chahi.core.entities import AnalysisContext, SourceCategory
from chahi.core.services.article_filter import ArticleFilterService

if TYPE_CHECKING:
    from chahi.core.entities import Article, SourceConfig
    from chahi.core.interfaces import (
        IConfigReader,
        ILLMClient,
        IMemoryManager,
        INewsFetcher,
    )

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════
# MAP Prompt — Phân tích cục bộ từng nhóm tài sản
# ═════════════════════════════════════════════════════════════

MAP_PROMPT: str = """\
Bạn là Chuyên gia Phân tích Tài chính. Input được cung cấp dưới dạng \
XML-like tags:
- `<category name="...">`
- `<article id="...">` với các field:
  `<title>`, `<source>`, `<published_at>`, `<summary>`.

Hãy đọc kỹ toàn bộ articles của nhóm **{category_name}** và viết BẢN TÓM TẮT gồm \
đúng 3-5 bullet points.

**YÊU CẦU PHÂN TÍCH**:
- Ưu tiên dữ kiện có tác động vĩ mô/cross-asset.
- Bỏ qua thông tin lặp lại hoặc ít giá trị.
- Chỉ tóm tắt DỮ KIỆN quan trọng, xu hướng, và con số nổi bật.
- Xác định tâm lý thị trường: tích cực / tiêu cực / trung lập.
- Ghi nhận yếu tố rủi ro hoặc catalyst tiềm năng.
- Viết súc tích, chuyên nghiệp, bằng tiếng Việt.
- KHÔNG viết heading hay tiêu đề, chỉ bullet points.
"""

# ═════════════════════════════════════════════════════════════
# REDUCE Prompt — Tổng hợp & Bản địa hóa Việt Nam
# ═════════════════════════════════════════════════════════════

REDUCE_PROMPT: str = """\
Bạn là Giám đốc Đầu tư (CIO) hàng đầu Việt Nam với hơn 20 năm kinh nghiệm \
điều phối danh mục tài sản lớn, đặc biệt am hiểu sâu sắc về VN-Index và \
nhóm cổ phiếu Ngân hàng. Bạn có tư duy quân sự: kỷ luật, tàn nhẫn khi sai, \
và luôn ghi nhật ký chiến trận để không bao giờ lặp lại sai lầm. Bạn cần tận dụng \
khả năng suy luận sâu kiểu Gemini 3.1 Pro: bám dữ liệu, liên kết tác động ngầm, \
không suy diễn vô căn cứ.

**NHIỆM VỤ**: Bạn nhận được 3 bản tóm tắt phân tích từ 3 nhóm tài sản \
(Dầu & Vĩ mô, Vàng, Crypto). Hãy TỔNG HỢP và SUY LUẬN TÁC ĐỘNG CHÉO \
giữa các lớp tài sản, sau đó viết Báo cáo Phân tích Vĩ mô chuyên sâu.

**ĐỊNH DẠNG INPUT**:
- `<analysis_date>`: ngày phân tích.
- `<hot_keywords>`: cụm từ nóng trích từ tin mới.
- `<map_summaries>`: kết quả MAP theo từng category.
- `<previous_lessons>`: bài học lịch sử trích từ Memory.

**TRỌNG TÂM SUY LUẬN**:
- Dòng tiền đang dịch chuyển thế nào giữa Dầu → Vàng → Crypto?
- Liên hệ giữa DXY, lãi suất Fed, lạm phát → tác động lên từng lớp tài sản.
- **BẢN ĐỊA HÓA BẮT BUỘC**: Đánh giá tác động của bức tranh vĩ mô toàn cầu \
lên Thị trường Chứng khoán Việt Nam (VN-Index).
- **NHÓM NGÂN HÀNG**: Phân tích chi tiết nhóm cổ phiếu ngân hàng VN — \
hưởng lợi hay chịu áp lực từ tỷ giá USD/VND, lạm phát, lãi suất quốc tế, \
tăng trưởng tín dụng, nợ xấu? Cơ hội hay rủi ro trong trung-dài hạn?

**ĐỐI CHIẾU QUÁ KHỨ** (BẮT BUỘC nếu có dữ liệu "[NHÌN LẠI QUÁ KHỨ]"):
- Đối chiếu tin tức hôm nay với nhận định quá khứ.
- Xác định: xu hướng tiếp diễn hay đảo chiều?
- Nếu nhận định cũ sai, phân tích tại sao và rút kinh nghiệm.
- Nếu nhận định cũ đúng, ghi nhận và dự báo bước tiếp theo.

**CẤU TRÚC BÁO CÁO BẮT BUỘC**:

# 📊 Báo cáo Phân tích Vĩ mô — {ngày}

## 1. 🛢️ Dầu & Kinh tế Vĩ mô
- Xu hướng giá dầu, cung-cầu, OPEC+.
- Fed, lạm phát, GDP, việc làm → tác động lên thị trường.

## 2. 🥇 Vàng & Kim loại quý
- Biến động giá vàng, dòng tiền ETF, nhu cầu trú ẩn.
- Tương quan DXY, lợi suất trái phiếu.

## 3. ₿ Crypto & Tài sản số
- Bitcoin, Ethereum, altcoin: xu hướng và dòng tiền.
- Tin pháp lý, ETF, institutional flow.

## 4. 🔄 Dòng tiền Xuyên lớp Tài sản
- Phân tích dịch chuyển dòng tiền: Dầu ↔ Vàng ↔ Crypto ↔ Cổ phiếu.
- So sánh nhận định phiên trước với thực tế hôm nay (nếu có).

## 5. 🇻🇳 Tác động lên Thị trường Việt Nam
- **VN-Index**: Dự báo xu hướng ngắn hạn dựa trên bức tranh vĩ mô.
- **Nhóm Ngân hàng**: Phân tích chi tiết rủi ro/cơ hội — \
tỷ giá USD/VND, lãi suất, tăng trưởng tín dụng, nợ xấu.
- **Các nhóm ngành khác**: Bất động sản, chứng khoán, xuất khẩu.
- Khuyến nghị hành động cho nhà đầu tư Việt Nam.

## 6. 📋 Tổng kết
- 3-5 bullet points chốt lại nhận định quan trọng nhất.
- Dự báo hướng đi cho ngày/tuần tiếp theo.
- Spotlight: nhóm ngân hàng — mua/giữ/bán?

## 7. 🧠 Sổ Tay Kinh Nghiệm
**BẮT BUỘC** — Đây là mục QUAN TRỌNG NHẤT cho sự tiến bộ của bạn.

Đọc kỹ phần [NHÌN LẠI QUÁ KHỨ] (nhật ký dự báo & bài học từ phiên trước \
của chính bạn). Đối chiếu với tin tức thực tế hôm nay và thực hiện:

- **Nếu dự báo SAI**: Tàn nhẫn tự kiểm điểm. Yếu tố nào bạn đã bỏ qua? \
Dòng tiền đã bẻ lái vì tin tức nào? Đúc kết thành 1 QUY TẮC PHÂN TÍCH MỚI \
(ví dụ: "Bài học: Khi có tin chiến tranh leo thang, bỏ qua yếu tố lạm phát — \
dòng tiền sẽ ưu tiên trú ẩn vào Vàng trước khi quay lại cổ phiếu").
- **Nếu dự báo ĐÚNG**: Ghi nhận yếu tố cốt lõi nào đã giúp dự báo chuẩn xác. \
Viết thành 1 QUY TẮC ĐỂ PHÁT HUY \
(ví dụ: "Kinh nghiệm: Khi DXY giảm liên tiếp 3 phiên + CPI hạ nhiệt → \
Vàng và Crypto đồng loạt tăng, VN-Index hưởng lợi qua nhóm xuất khẩu").
- **Nếu chưa có dữ liệu quá khứ** (lần đầu chạy): Ghi nhận 2-3 rủi ro/catalyst \
cần theo dõi cho phiên tiếp theo.

Format mỗi bài học:
> 📝 **Bài học #{số}**: [Mô tả ngắn gọn quy tắc]
> - Bối cảnh: [Tình huống dẫn tới bài học]
> - Quy tắc: [Quy tắc phân tích rút ra]

**QUY TẮC**:
- Suy luận dựa trên DỮ LIỆU thực tế, không suy đoán vô căn cứ.
- Ngôn ngữ chuyên nghiệp, súc tích, Giám đốc Đầu tư viết cho team.
- Trả lời hoàn toàn bằng tiếng Việt.
- Markdown chuẩn: heading, bullet points, bold/italic.
- **TUYỆT ĐỐI** bắt đầu phần tổng kết bằng chính xác dòng: `## 6. 📋 Tổng kết`
- **TUYỆT ĐỐI** bắt đầu phần kinh nghiệm bằng chính xác dòng: \
`## 7. 🧠 Sổ Tay Kinh Nghiệm`
"""

# ── Regex trích xuất Tổng kết + Sổ Tay Kinh Nghiệm ──
# Bắt từ "## Tổng kết" → lấy TOÀN BỘ nội dung đến hết file.
# Nội dung trả về sẽ chứa CẢ mục 6 (Tổng kết) VÀ mục 7 (Sổ Tay Kinh Nghiệm)
# để gửi nguyên khối vào IMemoryManager → phiên sau LLM đọc lại.
_SUMMARY_RE = re.compile(
    r"##\s*(?:\d+\.?\s*)?(?:📋\s*)?[Tt]ổng\s*[Kk]ết.*?\n(.*)",
    re.DOTALL,
)

# ── Ngân sách token tối đa cho user_content ──
# 28K tokens ≈ an toàn cho model 32K context (chừa chỗ cho system prompt)
# Vietnamese text: ~1 token / 1.7 chars (conservative)
_MAX_CONTEXT_TOKENS: int = 28_000
_CHARS_PER_TOKEN: float = 1.7  # Vietnamese heuristic

# ── Category names (tiếng Việt) cho MAP prompt ──
_CATEGORY_NAMES: dict[str, str] = {
    "oil_macro": "Dầu & Kinh tế Vĩ mô",
    "gold": "Vàng & Kim loại quý",
    "crypto": "Crypto & Tài sản số",
}


# ═════════════════════════════════════════════════════════════
# Use Case
# ═════════════════════════════════════════════════════════════


class GenerateMacroReportUseCase:
    """Use case: Tạo báo cáo phân tích vĩ mô (Map-Reduce + Memory).

    Pipeline Phase 8:
    1. **Retrieve**: Lấy nhận định ngày trước từ ``IMemoryManager``.
    2. Đọc cấu hình nguồn tin từ ``IConfigReader``.
    3. Fetch tin tức song song từ ``INewsFetcher``.
    4. **Filter**: Lọc keyword + khử trùng lặp (Jaccard).
    5. **Semantic Retrieve**: Query bài học liên quan theo hot keywords.
    6. **MAP**: Gọi LLM phân tích từng nhóm (3 calls).
    7. **REDUCE**: Gọi LLM tổng hợp → Báo cáo cuối + Vietnam focus.
    8. **Store**: Lưu phần Tổng kết vào Memory.

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
        memory_manager: IMemoryManager | None = None,
        article_filter: ArticleFilterService | None = None,
    ) -> None:
        self._config_reader = config_reader
        self._news_fetcher = news_fetcher
        self._llm_client = llm_client
        self._memory = memory_manager
        self._article_filter = article_filter or ArticleFilterService()

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
            logger.info("Bước 1/8: Truy xuất nhận định cũ từ Memory...")
            try:
                previous_context = self._memory.retrieve_last_context()
                if previous_context:
                    logger.info("  → Có nhận định cũ: %d chars", len(previous_context))
                else:
                    logger.info("  → Chưa có nhận định cũ (lần đầu chạy).")
            except Exception as exc:
                logger.warning("  → Lỗi đọc Memory, bỏ qua: %s", exc)
        else:
            logger.info("Bước 1/8: Memory Manager chưa được cấu hình, bỏ qua.")

        # ── Bước 2: Đọc config nguồn tin ──
        logger.info("Bước 2/8: Đọc cấu hình nguồn tin...")
        sources = self._config_reader.get_sources()

        # ── Bước 3: Fetch tin tức ──
        logger.info("Bước 3/8: Cào tin tức từ %d categories...", len(sources))
        context = self._fetch_all_news(sources, previous_context)
        if context.is_empty:
            logger.warning("Không có bài viết nào được fetch. Dừng pipeline.")
            return "⚠️ Không có tin tức nào để phân tích."

        logger.info(
            "Trước lọc: %d bài (Oil&Macro=%d, Gold=%d, Crypto=%d)",
            context.total_articles,
            len(context.oil_news),
            len(context.gold_news),
            len(context.crypto_news),
        )

        # ── Bước 4: Filter — Keyword + Jaccard Dedup ──
        logger.info("Bước 4/8: Lọc từ khóa & khử trùng lặp (Jaccard)...")
        context = self._article_filter.filter_context(context)
        if context.is_empty:
            logger.warning("Sau lọc không còn bài liên quan. Dừng pipeline.")
            return "⚠️ Không có tin tức phù hợp sau bước lọc từ khóa."
        logger.info(
            "Sau lọc: %d bài (Oil&Macro=%d, Gold=%d, Crypto=%d)",
            context.total_articles,
            len(context.oil_news),
            len(context.gold_news),
            len(context.crypto_news),
        )

        hot_keywords = self._article_filter.extract_hot_keywords(
            context, max_keywords=8
        )
        if hot_keywords:
            logger.info("Hot keywords: %s", ", ".join(hot_keywords))
        else:
            logger.info("Hot keywords: không trích được keyword nổi bật.")

        # ── Bước 5: Semantic retrieve từ Memory theo bối cảnh hiện tại ──
        semantic_context: str | None = None
        if self._memory is not None and hot_keywords:
            logger.info("Bước 5/8: Truy xuất bài học liên quan (semantic memory)...")
            try:
                semantic_context = self._memory.retrieve_related_context(
                    hot_keywords=hot_keywords,
                    max_results=3,
                )
                if semantic_context:
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
                "Bước 5/8: Bỏ qua semantic retrieval (không có memory/keywords)."
            )

        combined_context = self._merge_contexts(previous_context, semantic_context)

        # ── Bước 6: MAP — Phân tích cục bộ từng nhóm ──
        logger.info("Bước 6/8: MAP — Phân tích từng nhóm...")
        category_summaries = self._map_analyze_all(context)
        if not any(category_summaries.values()):
            logger.error("MAP thất bại hoàn toàn. Không có tóm tắt nào.")
            return "⚠️ Không thể phân tích tin tức (MAP failed)."

        # ── Bước 7: REDUCE — Tổng hợp & Bản địa hóa ──
        logger.info("Bước 7/8: REDUCE — Tổng hợp & Bản địa hóa VN...")
        reduce_input = self._format_reduce_input(
            category_summaries,
            previous_context=combined_context,
            hot_keywords=hot_keywords,
        )
        report = self._llm_client.analyze(
            system_prompt=REDUCE_PROMPT,
            user_content=reduce_input,
        )
        logger.info("✓ Báo cáo đã được tạo: %d chars", len(report))

        # ── Bước 8: STORE — Lưu nhận định mới vào Memory ──
        if self._memory is not None:
            logger.info("Bước 8/8: Lưu nhận định mới vào Memory...")
            try:
                summary = _extract_summary(report)
                self._memory.save_context(summary)
                logger.info("  → Đã lưu %d chars vào Memory.", len(summary))
            except Exception as exc:
                logger.warning("  → Lỗi lưu Memory, bỏ qua: %s", exc)
        else:
            logger.info("Bước 8/8: Memory Manager chưa được cấu hình, bỏ qua.")

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
        lines: list[str] = [f'<category name="{category_name}">']
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

    def _map_analyze_all(self, context: AnalysisContext) -> dict[str, str]:
        """Chạy MAP cho cả 3 nhóm tài sản.

        Tự động chọn chế độ:
        - Cloud LLM (Gemini): song song qua ThreadPoolExecutor.
        - Local LLM (LM Studio): tuần tự để tránh nghẽn queue.

        Args:
            context: AnalysisContext chứa tin tức.

        Returns:
            Dict mapping category_name → bản tóm tắt LLM.
        """
        tasks: list[tuple[str, list[Article]]] = [
            (_CATEGORY_NAMES["oil_macro"], list(context.oil_news)),
            (_CATEGORY_NAMES["gold"], list(context.gold_news)),
            (_CATEGORY_NAMES["crypto"], list(context.crypto_news)),
        ]

        results: dict[str, str] = {}

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

    @staticmethod
    def _format_reduce_input(
        summaries: dict[str, str],
        previous_context: str | None = None,
        hot_keywords: list[str] | None = None,
    ) -> str:
        """Format đầu vào cho REDUCE step.

        Ghép 3 bản tóm tắt MAP + previous_context thành
        user_content cho REDUCE prompt.

        Args:
            summaries: Dict category_name → tóm tắt từ MAP.
            previous_context: Nhận định phiên trước (Memory).
            hot_keywords: Danh sách từ khóa nóng từ dữ liệu phiên hiện tại.

        Returns:
            Chuỗi text formatted cho REDUCE prompt.
        """
        today = date.today().strftime("%d/%m/%Y")
        sections: list[str] = [f"<analysis_date>{today}</analysis_date>\n"]

        if hot_keywords:
            sections.append("<hot_keywords>")
            sections.append(", ".join(hot_keywords))
            sections.append("</hot_keywords>\n")

        sections.append("<map_summaries>")
        ordered_categories = [
            _CATEGORY_NAMES["oil_macro"],
            _CATEGORY_NAMES["gold"],
            _CATEGORY_NAMES["crypto"],
        ]
        for category_name in ordered_categories:
            summary = summaries.get(category_name, "")
            sections.append(f'  <category name="{category_name}">')
            sections.append(
                summary.strip() if summary.strip() else "(Không có dữ liệu)"
            )
            sections.append("  </category>")
        sections.append("</map_summaries>\n")

        if previous_context:
            sections.append("<previous_lessons>")
            sections.append("🔄 [NHÌN LẠI QUÁ KHỨ]")
            sections.append(previous_context.strip())
            sections.append("</previous_lessons>")

        result = "\n".join(sections)

        # ── Token budget guard (Vietnamese-safe heuristic) ──
        estimated_tokens = int(len(result) / _CHARS_PER_TOKEN)
        if estimated_tokens > _MAX_CONTEXT_TOKENS:
            # Cắt tại character limit tương đương
            max_chars = int(_MAX_CONTEXT_TOKENS * _CHARS_PER_TOKEN)
            logger.warning(
                "REDUCE input quá dài (~%d tokens > %d). Cắt bớt.",
                estimated_tokens,
                _MAX_CONTEXT_TOKENS,
            )
            result = result[:max_chars] + "\n\n⚠️ (Đã cắt bớt do giới hạn token)"

        return result

    # ── Fetch tin tức ─────────────────────────────────────────

    _MAX_WORKERS: int = 5  # concurrent RSS fetchers

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
        oil_news: list[Article] = []
        gold_news: list[Article] = []
        crypto_news: list[Article] = []

        category_map: dict[SourceCategory, list[Article]] = {
            SourceCategory.OIL_MACRO: oil_news,
            SourceCategory.GOLD: gold_news,
            SourceCategory.CRYPTO: crypto_news,
        }

        # Chuẩn bị danh sách tasks
        tasks: list[tuple[SourceCategory, SourceConfig]] = []
        for category, source_list in sources.items():
            for source in source_list:
                tasks.append((category, source))

        # Fetch song song
        with ThreadPoolExecutor(max_workers=self._MAX_WORKERS) as pool:
            future_to_source = {
                pool.submit(
                    self._news_fetcher.fetch_news,
                    url=source.url,
                    limit=10,
                ): (category, source)
                for category, source in tasks
            }

            for future in as_completed(future_to_source):
                category, source = future_to_source[future]
                target = category_map.get(category, [])

                try:
                    articles = future.result()
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
            oil_news=oil_news,
            gold_news=gold_news,
            crypto_news=crypto_news,
            previous_context=previous_context,
        )


def _extract_summary(report: str) -> str:
    """Trích xuất phần Tổng kết + Sổ Tay Kinh Nghiệm từ báo cáo.

    Tìm section ``## 6. 📋 Tổng kết`` và lấy TOÀN BỘ nội dung phía sau,
    bao gồm cả ``## 7. 🧠 Sổ Tay Kinh Nghiệm``. Khối này được gửi
    nguyên vẹn vào ``IMemoryManager`` để phiên sau LLM đọc lại và
    tự đối chiếu (Self-Reflection).

    Nếu không tìm được, lưu 500 ký tự cuối cùng của report.

    Args:
        report: Nội dung báo cáo Markdown đầy đủ.

    Returns:
        Phần tổng kết + kinh nghiệm đã trích xuất.
    """
    match = _SUMMARY_RE.search(report)
    if match:
        return match.group(1).strip()

    # Fallback: lấy phần cuối report
    return report[-500:].strip() if len(report) > 500 else report.strip()
