"""Use Cases — Application logic ghép nối tất cả components.

Module này chứa các use cases (application services) là nơi
orchestrate workflow chính của ChaHi. Mỗi use case nhận
dependencies qua constructor (DI) và expose một method ``execute()``.

Quy tắc:
    - Use case chỉ phụ thuộc vào core/ interfaces và entities.
    - Không import trực tiếp infrastructure modules.
    - Prompts được định nghĩa tại đây vì chúng là application logic.

Pipeline Phase 7 (Map-Reduce + Feedback Loop):
    1. **Retrieve**: Lấy nhận định cũ từ Memory.
    2. **Fetch**: Cào tin tức song song từ RSS.
    3. **MAP**: Gọi LLM phân tích từng nhóm (Dầu, Vàng, Crypto) song song.
    4. **REDUCE**: Gọi LLM tổng hợp 3 bản tóm tắt + Memory → Báo cáo cuối.
    5. **Store**: Lưu phần Tổng kết vào Memory.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import TYPE_CHECKING

from chahi.core.entities import AnalysisContext, SourceCategory

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
Bạn là Chuyên gia Phân tích Tài chính. Đọc kỹ danh sách tin tức \
của nhóm **{category_name}** dưới đây và viết BẢN TÓM TẮT gồm \
đúng 3-5 ý chính (bullet points).

**YÊU CẦU**:
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
và luôn ghi nhật ký chiến trận để không bao giờ lặp lại sai lầm.

**NHIỆM VỤ**: Bạn nhận được 3 bản tóm tắt phân tích từ 3 nhóm tài sản \
(Dầu & Vĩ mô, Vàng, Crypto). Hãy TỔNG HỢP và SUY LUẬN TÁC ĐỘNG CHÉO \
giữa các lớp tài sản, sau đó viết Báo cáo Phân tích Vĩ mô chuyên sâu.

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
# 100_000 chars ≈ 33_000 tokens — hardware mạnh, không cần chắt bóp
_MAX_CONTEXT_CHARS: int = 100_000

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

    Pipeline Phase 7:
    1. **Retrieve**: Lấy nhận định ngày trước từ ``IMemoryManager``.
    2. Đọc cấu hình nguồn tin từ ``IConfigReader``.
    3. Fetch tin tức song song từ ``INewsFetcher``.
    4. **MAP**: Gọi LLM phân tích từng nhóm (3 calls song song).
    5. **REDUCE**: Gọi LLM tổng hợp → Báo cáo cuối + Vietnam focus.
    6. **Store**: Lưu phần Tổng kết vào Memory.

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
    ) -> None:
        self._config_reader = config_reader
        self._news_fetcher = news_fetcher
        self._llm_client = llm_client
        self._memory = memory_manager

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

        # ── Bước 4: MAP — Phân tích cục bộ từng nhóm (song song) ──
        logger.info("Bước 4/7: MAP — Phân tích từng nhóm song song...")
        category_summaries = self._map_analyze_all(context)

        if not any(category_summaries.values()):
            logger.error("MAP thất bại hoàn toàn. Không có tóm tắt nào.")
            return "⚠️ Không thể phân tích tin tức (MAP failed)."

        # ── Bước 5: REDUCE — Tổng hợp & Bản địa hóa ──
        logger.info("Bước 5/7: REDUCE — Tổng hợp & Bản địa hóa VN...")
        reduce_input = self._format_reduce_input(category_summaries, previous_context)
        report = self._llm_client.analyze(
            system_prompt=REDUCE_PROMPT,
            user_content=reduce_input,
        )

        logger.info("✓ Báo cáo đã được tạo: %d chars", len(report))

        # ── Bước 6: STORE — Lưu nhận định mới vào Memory ──
        if self._memory is not None:
            logger.info("Bước 6/7: Lưu nhận định mới vào Memory...")
            try:
                summary = _extract_summary(report)
                self._memory.save_context(summary)
                logger.info("  → Đã lưu %d chars vào Memory.", len(summary))
            except Exception as exc:
                logger.warning("  → Lỗi lưu Memory, bỏ qua: %s", exc)
        else:
            logger.info("Bước 6/7: Memory Manager chưa được cấu hình, bỏ qua.")

        return report

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

        # Format articles → text
        lines: list[str] = []
        for i, article in enumerate(articles, 1):
            pub_date = article.published_date.strftime("%d/%m/%Y %H:%M")
            lines.append(f"--- Tin #{i} ---")
            lines.append(f"Tiêu đề : {article.title}")
            lines.append(f"Nguồn   : {article.source_name}")
            lines.append(f"Ngày    : {pub_date}")
            lines.append(f"Tóm tắt : {article.summary}")
            lines.append("")

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
        """Chạy MAP song song cho cả 3 nhóm tài sản.

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

        return results

    # ── REDUCE: Format input ─────────────────────────────────

    @staticmethod
    def _format_reduce_input(
        summaries: dict[str, str],
        previous_context: str | None = None,
    ) -> str:
        """Format đầu vào cho REDUCE step.

        Ghép 3 bản tóm tắt MAP + previous_context thành
        user_content cho REDUCE prompt.

        Args:
            summaries: Dict category_name → tóm tắt từ MAP.
            previous_context: Nhận định phiên trước (Memory).

        Returns:
            Chuỗi text formatted cho REDUCE prompt.
        """
        today = date.today().strftime("%d/%m/%Y")
        sections: list[str] = [f"📅 Ngày phân tích: {today}\n"]

        # ── Previous Context (Nhật ký Self-Reflection) ──
        if previous_context:
            sections.append("=" * 50)
            sections.append(
                "🔄 [NHÌN LẠI QUÁ KHỨ] — Nhật ký dự báo"
                " & Bài học của bạn từ phiên trước:"
            )
            sections.append("=" * 50)
            sections.append(previous_context)
            sections.append("")

        # ── MAP Summaries ──
        sections.append("=" * 50)
        sections.append("📊 KẾT QUẢ PHÂN TÍCH CỤC BỘ (MAP)")
        sections.append("=" * 50)

        for name, summary in summaries.items():
            sections.append(f"\n### 📰 {name}")
            if summary.strip():
                sections.append(summary.strip())
            else:
                sections.append("(Không có dữ liệu)")
            sections.append("")

        result = "\n".join(sections)

        # ── Token budget guard ──
        if len(result) > _MAX_CONTEXT_CHARS:
            logger.warning(
                "REDUCE input quá dài (%d chars > %d). Cắt bớt.",
                len(result),
                _MAX_CONTEXT_CHARS,
            )
            result = (
                result[:_MAX_CONTEXT_CHARS] + "\n\n⚠️ (Đã cắt bớt do giới hạn token)"
            )

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
