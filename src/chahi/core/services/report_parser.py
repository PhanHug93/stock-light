"""Report Parser — trích xuất cấu trúc từ output LLM.

Tách khỏi use_cases.py để tuân thủ SRP:
use_cases chỉ orchestrate, parser chỉ parse.
"""

from __future__ import annotations

import re

# Bắt từ "## Tổng kết" → lấy TOÀN BỘ nội dung đến hết file.
# Nội dung trả về sẽ chứa CẢ mục 6 (Tổng kết) VÀ mục 7 (Sổ Tay Kinh Nghiệm)
# để gửi nguyên khối vào IMemoryManager → phiên sau LLM đọc lại.
_SUMMARY_RE = re.compile(
    r"##\s*(?:\d+\.?\s*)?(?:📋\s*)?[Tt]ổng\s*[Kk]ết.*?\n(.*)",
    re.DOTALL,
)


def extract_summary(report: str) -> str:
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
