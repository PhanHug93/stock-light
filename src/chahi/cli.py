"""ChaHi CLI — Entry point và orchestration.

Module này chứa logic CLI: parse arguments, load config,
và hiển thị summary. Pipeline phân tích sẽ được thêm sau.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chahi import __version__
from chahi.infrastructure.config.yaml_config_reader import YamlConfigReader

_BANNER = r"""
   _____ _           _    _ _
  / ____| |         | |  | (_)
 | |    | |__   __ _| |__| |_
 | |    | '_ \ / _` |  __  | |
 | |____| | | | (_| | |  | | |
  \_____|_| |_|\__,_|_|  |_|_|
"""


def _build_parser() -> argparse.ArgumentParser:
    """Tạo ArgumentParser cho ChaHi CLI.

    Returns:
        ArgumentParser đã được cấu hình đầy đủ.
    """
    parser = argparse.ArgumentParser(
        prog="chahi",
        description="ChaHi — Phân tích tin tức tài chính vĩ mô qua Local LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Đường dẫn tới file cấu hình YAML (mặc định: config.yaml).",
        metavar="PATH",
    )
    return parser


def _print_config_summary(config_path: Path) -> None:
    """Load config và in tóm tắt ra stdout.

    Args:
        config_path: Đường dẫn tới file YAML cấu hình.
    """
    reader = YamlConfigReader(config_path=config_path)

    # ── Sources ──
    sources = reader.get_sources()
    total = sum(len(v) for v in sources.values())
    print(f"  Config    : {config_path.resolve()}")
    print(f"  Sources   : {total} nguồn")
    for category, source_list in sources.items():
        print(f"              └─ {category}: {len(source_list)}")

    # ── LLM ──
    llm = reader.get_llm_settings()
    print(f"  LLM       : {llm.api_base} (model: {llm.model_name})")
    print()


def main() -> None:
    """CLI entry point chính.

    Parse arguments, load config, hiển thị summary.
    Thoát với exit code 1 nếu có lỗi.
    """
    parser = _build_parser()
    args = parser.parse_args()

    print(_BANNER)
    print(f"  ChaHi v{__version__}")
    print("  ─────────────────────────────────")

    try:
        _print_config_summary(args.config)
    except FileNotFoundError as exc:
        print(f"\n  ✗ Lỗi: {exc}", file=sys.stderr)
        print(
            "  → Tạo file config.yaml từ config.example.yaml để bắt đầu.",
            file=sys.stderr,
        )
        sys.exit(1)
    except ValueError as exc:
        print(f"\n  ✗ Config không hợp lệ: {exc}", file=sys.stderr)
        sys.exit(1)

    print("  ✓ Config loaded thành công. Pipeline chưa sẵn sàng (Phase 1+2).")
