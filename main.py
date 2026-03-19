"""ChaHi — Main entry point: chạy pipeline phân tích vĩ mô.

Khởi tạo toàn bộ dependencies (Config, Fetcher, LLM Client),
inject vào Use Case, chạy pipeline, và lưu báo cáo Markdown.

Logging có thể cấu hình qua config.yaml (section ``logging``).

Usage:
    python main.py
    python main.py --config path/to/config.yaml
    python main.py --output-dir ./my-reports
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from chahi.core.use_cases import GenerateMacroReportUseCase
from chahi.infrastructure.config.yaml_config_reader import YamlConfigReader
from chahi.infrastructure.llm.llm_factory import create_llm_client
from chahi.infrastructure.memory.memory_factory import create_memory_manager
from chahi.infrastructure.notifiers.notifier_factory import create_notification_manager
from chahi.infrastructure.rss.rss_fetcher import RSSNewsFetcher

# ── Constants ────────────────────────────────────────────────

_DEFAULT_OUTPUT_DIR = "./reports"

_BANNER = r"""
   _____ _           _    _ _
  / ____| |         | |  | (_)
 | |    | |__   __ _| |__| |_
 | |    | '_ \ / _` |  __  | |
 | |____| | | | (_| | |  | | |
  \_____|_| |_|\__,_|_|  |_|_|
"""


# ── Logging Setup ───────────────────────────────────────────


def _setup_logging(config_path: Path) -> None:
    """Thiết lập logging từ config YAML hoặc dùng mặc định.

    Đọc section ``logging`` từ config file:
        - ``level``: DEBUG, INFO, WARNING, ERROR (mặc định: INFO)
        - ``file``: Đường dẫn log file (optional, mặc định: chỉ console)
        - ``format``: Log format string (optional)

    Args:
        config_path: Đường dẫn tới file config YAML.
    """
    # Defaults
    log_level = "INFO"
    log_file: str | None = None
    log_format = "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s"
    date_format = "%H:%M:%S"

    # Đọc config nếu có
    try:
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                raw: dict[str, Any] = yaml.safe_load(f) or {}
            log_config = raw.get("logging", {})
            if isinstance(log_config, dict):
                log_level = str(log_config.get("level", log_level)).upper()
                log_file = log_config.get("file")
                log_format = str(log_config.get("format", log_format))
    except Exception:
        pass  # Nếu lỗi, dùng defaults

    # ── Handlers ──
    handlers: list[logging.Handler] = []

    # Console handler (luôn có)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    handlers.append(console)

    # File handler (optional)
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        handlers.append(file_handler)

    # Apply
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        handlers=handlers,
    )


logger = logging.getLogger("chahi.main")


# ── CLI ──────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    """Tạo ArgumentParser."""
    parser = argparse.ArgumentParser(description="ChaHi — Macro Report Generator")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Đường dẫn tới file config YAML (mặc định: config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Thư mục lưu báo cáo (mặc định: {_DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--notify-only",
        action="store_true",
        default=False,
        help="Gửi lại báo cáo mới nhất qua các kênh đã bật (không chạy pipeline).",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        default=False,
        help="Chỉ gửi notification qua Telegram.",
    )
    parser.add_argument(
        "--discord",
        action="store_true",
        default=False,
        help="Chỉ gửi notification qua Discord.",
    )
    return parser


def _validate_output_dir(output_dir: Path) -> Path:
    """Validate và resolve output directory.

    Chống path traversal: đảm bảo output nằm trong project root.

    Args:
        output_dir: Đường dẫn output.

    Returns:
        Path đã resolve và validate.

    Raises:
        ValueError: Khi path trỏ ra ngoài project root.
    """
    project_root = Path.cwd().resolve()
    resolved = (project_root / output_dir).resolve()

    if not str(resolved).startswith(str(project_root)):
        raise ValueError(
            f"Output directory phải nằm trong project root. "
            f"Nhận được: '{output_dir}' → resolve thành '{resolved}'. "
            f"Project root: '{project_root}'"
        )

    return resolved


def _save_report(content: str, output_dir: Path) -> Path:
    """Lưu báo cáo Markdown ra file.

    Args:
        content: Nội dung báo cáo Markdown.
        output_dir: Thư mục output đã validate.

    Returns:
        Path tuyệt đối tới file đã lưu.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    today = date.today().strftime("%Y%m%d")
    filename = f"report_{today}.md"
    file_path = output_dir / filename

    file_path.write_text(content, encoding="utf-8")
    return file_path.resolve()


def _get_latest_report(output_dir: Path) -> tuple[Path, str] | None:
    """Tìm và đọc báo cáo mới nhất từ thư mục reports.

    Quét tất cả file ``report_*.md`` trong output_dir, sắp xếp theo tên
    (YYYYMMDD → tên lớn nhất = ngày mới nhất), rồi đọc nội dung.

    Args:
        output_dir: Thư mục chứa các file báo cáo.

    Returns:
        Tuple ``(path, content)`` của báo cáo mới nhất.
        ``None`` nếu chưa có báo cáo nào.
    """
    if not output_dir.exists():
        return None

    reports = sorted(output_dir.glob("report_*.md"), reverse=True)
    if not reports:
        return None

    latest = reports[0]
    content = latest.read_text(encoding="utf-8")
    logger.info("Đọc báo cáo mới nhất: %s (%d chars)", latest.name, len(content))
    return latest.resolve(), content


def _send_notifications(
    config_reader: YamlConfigReader,
    report_content: str,
    channels: list[str] | None = None,
) -> None:
    """Gửi báo cáo tới các kênh thông báo đã enabled.

    Đọc notification settings từ IConfigReader, khởi tạo
    NotificationManager (Composite), và gửi report.
    Fail-safe: mọi exception đều được bắt và log.

    Args:
        config_reader: Config reader đã khởi tạo.
        report_content: Nội dung báo cáo Markdown cần gửi.
        channels: Danh sách kênh cần gửi (["telegram"], ["discord"], hoặc
                  None = tất cả kênh đã enabled).
    """
    try:
        all_settings = config_reader.get_notification_settings()

        # Lọc kênh nếu có chỉ định
        if channels:
            all_settings = [s for s in all_settings if s.type in channels]
            logger.info("Gửi chỉ qua: %s", ", ".join(channels))

        manager = create_notification_manager(all_settings)

        if manager.client_count > 0:
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            title = f"ChaHi Report {now}"
            success = manager.send_report(title, report_content)
            if success:
                logger.info(
                    "✓ Đã gửi báo cáo qua %d kênh",
                    manager.client_count,
                )
            else:
                logger.warning("✗ Có kênh gửi thất bại (xem log chi tiết)")
        else:
            logger.info("Không có kênh thông báo nào được bật.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lỗi gửi thông báo: %s", exc)


def _print_preview(report_content: str) -> None:
    """In preview báo cáo ra console."""
    preview = report_content[:500]
    print(f"\n{'─' * 50}")
    print("📄 Preview báo cáo:")
    print(f"{'─' * 50}")
    print(preview)
    if len(report_content) > 500:
        print(f"\n... (còn {len(report_content) - 500} chars)")
    print(f"{'─' * 50}\n")


def main() -> None:
    """Entry point chính — chạy toàn bộ pipeline."""
    parser = _build_parser()
    args = parser.parse_args()

    # ── Setup logging từ config ──
    _setup_logging(args.config)

    print(_BANNER)
    logger.info("ChaHi — Macro Report Generator")
    logger.info("=" * 50)

    output_dir = args.output_dir or Path(_DEFAULT_OUTPUT_DIR)

    # Xác định kênh gửi (dùng chung cho cả 2 mode)
    channels: list[str] | None = None
    if args.telegram or args.discord:
        channels = []
        if args.telegram:
            channels.append("telegram")
        if args.discord:
            channels.append("discord")

    # ══════════════════════════════════════════════════
    # Mode: --notify-only → gửi lại báo cáo mới nhất
    # ══════════════════════════════════════════════════
    if args.notify_only:
        logger.info("Mode: --notify-only (gửi lại báo cáo mới nhất)")

        try:
            validated_dir = _validate_output_dir(output_dir)
        except ValueError as exc:
            logger.error("✗ %s", exc)
            sys.exit(1)

        result = _get_latest_report(validated_dir)
        if result is None:
            logger.error("✗ Không tìm thấy báo cáo nào trong: %s", validated_dir)
            sys.exit(1)

        report_path, report_content = result
        logger.info("Báo cáo mới nhất: %s", report_path)

        config_reader = YamlConfigReader(config_path=args.config)
        _send_notifications(config_reader, report_content, channels=channels)
        _print_preview(report_content)
        return

    # ══════════════════════════════════════════════════
    # Mode: Chạy pipeline đầy đủ
    # ══════════════════════════════════════════════════

    # ── 1. Load config ──
    try:
        config_reader = YamlConfigReader(config_path=args.config)
        llm_settings = config_reader.get_llm_settings()
        logger.info("Config loaded: %s", args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("✗ Lỗi config: %s", exc)
        sys.exit(1)

    # ── 2. Khởi tạo dependencies (Strategy Pattern + Memory) ──
    news_fetcher = RSSNewsFetcher(source_name="RSS")
    llm_client = create_llm_client(settings=llm_settings)

    memory_settings = config_reader.get_memory_settings()
    memory_manager = create_memory_manager(settings=memory_settings)

    # ── 3. Khởi tạo & chạy Use Case ──
    use_case = GenerateMacroReportUseCase(
        config_reader=config_reader,
        news_fetcher=news_fetcher,
        llm_client=llm_client,
        memory_manager=memory_manager,
    )

    try:
        logger.info("─" * 50)
        report_content = use_case.execute()
        logger.info("─" * 50)
    except ConnectionError as exc:
        logger.error("✗ %s", exc)
        sys.exit(1)
    except RuntimeError as exc:
        logger.error("✗ Lỗi LLM: %s", exc)
        sys.exit(1)

    # ── 4. Validate & lưu báo cáo ──
    try:
        validated_dir = _validate_output_dir(output_dir)
    except ValueError as exc:
        logger.error("✗ %s", exc)
        sys.exit(1)

    report_path = _save_report(report_content, validated_dir)
    logger.info("=" * 50)
    logger.info("✓ Báo cáo đã lưu: %s", report_path)
    logger.info("=" * 50)

    # ── 5. Gửi thông báo (Telegram / Discord) ──
    _send_notifications(config_reader, report_content, channels=channels)

    # ── Print preview ──
    _print_preview(report_content)


if __name__ == "__main__":
    main()
