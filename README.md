# ChaHi — Macro Report Generator

CLI phân tích tin tức vĩ mô từ RSS (Oil/Macro, Gold, Crypto), tạo báo cáo Markdown và gửi Telegram/Discord.

Phiên bản: **v0.0.5**

## Mục tiêu

- Cào RSS theo nhiều nguồn và nhiều nhóm tài sản.
- Chuẩn hóa dữ liệu đầu vào cho LLM theo pipeline MAP/REDUCE.
- Hỗ trợ nhiều provider: `gemini` (mặc định), `openai`, `lm_studio`.
- Có chế độ xuất input độc lập để request ở hệ thống khác.

## Cài đặt nhanh

```bash
git clone <repo-url>
cd stock-light

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock
pip install -e ".[dev]"
```

Tạo file cấu hình:

```bash
cp config.example.yaml config.yaml
```

## Cấu hình tối thiểu (`config.yaml`)

```yaml
sources:
  oil_macro:
    - name: "Reuters Business"
      url: "https://feeds.reuters.com/reuters/businessNews"
      type: rss
  gold:
    - name: "Kitco Gold News"
      url: "https://www.kitco.com/rss/gold.xml"
      type: rss
  crypto:
    - name: "CoinDesk"
      url: "https://www.coindesk.com/arc/outboundfeeds/rss/"
      type: rss

llm_settings:
  provider: "gemini"            # gemini | openai | lm_studio
  api_key: ""                   # bắt buộc cho gemini/openai
  model_name: "gemini-3.1-pro"
  temperature: 0.1
  timeout: 120

memory:
  type: "file"                  # file | mcp
```

Lưu ý khi dùng OpenAI:

- Cần `llm_settings.api_key` hoặc `--api-key`/`OPENAI_API_KEY`.
- ChatGPT Plus không tự cấp API credits cho OpenAI API.

## Lệnh chạy thực tế

Chạy mặc định:

```bash
python main.py
```

Override provider từ CLI (không cần sửa file config):

```bash
# OpenAI
python main.py --provider openai --api-key "$OPENAI_API_KEY" --model gpt-5.4

# Gemini (mặc định)
python main.py --provider gemini --api-key "$GEMINI_API_KEY" --model gemini-3.1-pro

# LM Studio
python main.py --provider lm_studio --api-base http://localhost:1234/v1 --api-key lm-studio --model qwen2.5-7b-instruct
```

Chế độ dump input để debug/fallback:

```bash
# Vẫn chạy provider, nhưng dump input trước khi gọi
python main.py --dump-llm-input-dir ./llm_inputs --no-memory-store

# Chỉ dump input, không gọi provider (độc lập hoàn toàn)
python main.py --dump-llm-input-only --dump-llm-input-dir ./llm_inputs
```

Giữ ổn định output cùng input (mặc định bật, ngưỡng 15%):

```bash
python main.py --max-output-drift-pct 15
python main.py --disable-output-drift-guard
```

Notification:

```bash
# Chạy pipeline + gửi theo kênh chỉ định
python main.py --telegram
python main.py --discord

# Chỉ gửi lại báo cáo mới nhất
python main.py --notify-only
python main.py --notify-only --telegram
python main.py --notify-only --discord

# Gửi kèm file dump .md lên Discord
python main.py --discord --dump-llm-input-dir ./llm_inputs --discord-attach-dump
```

## Kết quả đầu ra

- Báo cáo: `reports/report_YYYYMMDD.md`
- LLM input dump: `llm_inputs/llm_input_XXX_<timestamp>.md`
- Cache ổn định output: `.cache/output_stability/`

## Kiến trúc ngắn gọn

- `src/chahi/core`: entities, interfaces, use case orchestration.
- `src/chahi/infrastructure/rss`: crawler async an toàn (semaphore, jitter, retry/backoff, deep scrape fallback).
- `src/chahi/infrastructure/llm`: clients cho Gemini/OpenAI/LM Studio + wrappers dump/capture/stability.
- `src/chahi/infrastructure/memory`: file memory hoặc MCP memory.
- `src/chahi/infrastructure/notifiers`: Telegram, Discord.

## Kiểm tra trước khi push

```bash
.venv/bin/ruff check src tests
.venv/bin/pytest -q
```

## Troubleshooting nhanh

- `OpenAI api_key must be set`: thiếu `api_key` trong config/CLI/env.
- `429 Rate limit`/`insufficient_quota`: giảm tần suất gọi hoặc nạp quota API.
- MCP không kết nối: tạm dùng `memory.type: file`.
- RSS nguồn lỗi hoặc chết feed: thay URL trong `sources`.
- Không muốn làm bẩn memory khi debug: dùng `--no-memory-store` hoặc `--dump-llm-input-only`.

## License

MIT
