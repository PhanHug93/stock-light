# 📊 ChaHi — Macro Report Generator

CLI tool phân tích tin tức tài chính vĩ mô (**Dầu · Vàng · Crypto**) thông qua Local LLM hoặc Google Gemini. Tự động cào tin RSS, tổng hợp dữ liệu, và xuất báo cáo Markdown chuyên sâu — chạy hoàn toàn trên máy cá nhân.

```
   _____ _           _    _ _
  / ____| |         | |  | (_)
 | |    | |__   __ _| |__| |_
 | |    | '_ \ / _` |  __  | |
 | |____| | | | (_| | |  | | |
  \_____|_| |_|\__,_|_|  |_|_|
```

---

## ✨ Tính năng

| Tính năng | Mô tả |
|-----------|--------|
| 🛢️ **Dầu & Vĩ mô** | Cào tin Reuters, CNBC — đánh giá cung-cầu, OPEC+, lạm phát |
| 🥇 **Vàng** | Theo dõi DXY, lợi suất trái phiếu, dòng tiền ETF |
| ₿ **Crypto** | BTC/ETH, tin pháp lý, dòng tiền tổ chức |
| 🔄 **Long-term Memory** | Đối chiếu nhận định qua các phiên (File / MCP Server) |
| 🤖 **Đa nền tảng LLM** | LM Studio (local) hoặc Google Gemini (cloud) |
| 🔌 **Dual-Protocol MCP** | Custom REST hoặc JSON-RPC SSE bridge chuẩn |
| 📝 **Markdown Output** | Báo cáo 6 mục với cấu trúc chuyên sâu |
| 📢 **Notifications** | Tự động gửi báo cáo qua Telegram & Discord (Composite Pattern) |

---

## 🏗️ Kiến trúc

```
Clean Architecture + SOLID + Dependency Injection

┌──────────────────────────────────────────────────────────┐
│                        main.py                           │  ← Entry point CLI
├──────────────────────────────────────────────────────────┤
│                     core/ (Domain)                       │
│  entities.py  │  interfaces.py  │  use_cases.py          │
├──────────────────────────────────────────────────────────┤
│                 infrastructure/ (I/O)                    │
│  config/   │  llm/     │  rss/    │ memory/  │ notifiers/│
│  YamlConfig│ LMStudio  │ RSS     │ File     │ Telegram  │
│            │ Gemini    │ Fetcher │ MCP(HTTP)│ Discord   │
│            │           │         │          │ Manager   │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 Cài đặt

### Yêu cầu

- **Python** ≥ 3.11
- **LM Studio** (nếu dùng local) hoặc **Gemini API Key** (nếu dùng cloud)

### Bước 1 — Clone & tạo môi trường

```bash
git clone <repo-url>
cd stock-light

python3 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
```

### Bước 2 — Cài dependencies

```bash
# Cài từ lockfile (khuyến nghị — reproducible builds)
pip install -r requirements.lock
pip install -e ".[dev]"
```

> **Regenerate lockfile** khi thêm/đổi dependency:
> ```bash
> pip-compile pyproject.toml -o requirements.lock --strip-extras --no-header
> ```

| Package | Vai trò |
|---------|---------|
| `feedparser` | Parse RSS feeds |
| `beautifulsoup4` | Xử lý HTML rác trong RSS |
| `requests` | HTTP client (RSS + MCP) |
| `openai` | Kết nối LM Studio (OpenAI-compatible) + OpenAI API |
| `google-genai` | Kết nối Google Gemini |
| `pyyaml` | Đọc config YAML |
| `tiktoken` | Đếm token chính xác (LLM budget) |
| `trafilatura` | Deep Scraper — bóc tách full-text |
| `filelock` | File locking — an toàn concurrent |

### Bước 3 — Tạo config

```bash
cp config.example.yaml config.yaml
```

---

## ⚙️ Cấu hình (`config.yaml`)

### 📰 Nguồn tin RSS

Thêm/xóa nguồn tùy ý theo 3 nhóm:

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
```

### 🤖 LLM Provider

Mặc định hệ thống dùng **Gemini**. Có thể đổi sang **OpenAI** hoặc **LM Studio**:

```yaml
llm_settings:
  provider: "gemini"                 # "gemini" | "openai" | "lm_studio"
  api_key: "AIza..."
  model_name: "gemini-2.0-flash"
  temperature: 0.1
  timeout: 120                        # Tăng 300-600 cho model 70B+
```

- OpenAI: `provider: "openai"`, `api_key: "sk-..."`, `model_name: "gpt-5.4"`
- LM Studio: `provider: "lm_studio"`, `api_base: "http://localhost:1234/v1"`

### 🧠 Memory (Bộ nhớ dài hạn)

Lưu nhận định cũ → đối chiếu với tin mới mỗi phiên:

```yaml
memory:
  # ── Option A: File (mặc định, hoạt động ngay) ──
  type: "file"

  # ── Option B: MCP Server ──
  # type: "mcp"
  # url: "http://localhost:8080"
  # workspace_path: "/path/to/project"
  # protocol: "jsonrpc"              # "rest" | "jsonrpc"
```

| Protocol | Endpoint | Format | Khi nào dùng |
|----------|----------|--------|-------------|
| `rest` | `POST /call-tool` | Custom JSON | MCP server có REST API riêng |
| `jsonrpc` | `GET /sse` → `POST /message` | JSON-RPC 2.0 | MCP server bật SSE bridge mode |

### 📋 Logging

```yaml
logging:
  level: "INFO"                # DEBUG | INFO | WARNING | ERROR
  # file: "chahi.log"          # Uncomment để log ra file
```

### 📢 Notifications (Telegram / Discord)

Tự động gửi báo cáo sau khi phân tích xong:

```yaml
notifications:
  telegram:
    enabled: true
    bot_token: "123456:ABC-DEF"    # Lấy từ @BotFather
    chat_id: "-1001234567890"      # Group ID (số âm) — lấy qua @RawDataBot
  discord:
    enabled: true
    webhook_url: "https://discord.com/api/webhooks/..."  # Server Settings → Integrations
```

> **Lưu ý:** Telegram Chat ID của group luôn là **số âm** (vd: `-1001234567890`). Thêm `@RawDataBot` vào group để lấy ID.

---

## 🎮 Sử dụng

### Chuẩn bị LM Studio (nếu dùng local)

1. Mở **LM Studio** → tải model (khuyến nghị: Qwen 2.5 7B / Llama 3 8B)
2. Tab **Local Server** → **▶ Start Server** (port `1234`)
3. Kéo **Context Length** lên `16384+` (RAM ≥ 32GB nên dùng `32768`)

### Chạy phân tích

```bash
python main.py                                    # Config mặc định
python main.py --config path/to/config.yaml       # Config tùy chỉnh
python main.py --output-dir ./my-reports           # Output tùy chỉnh
```

### Control command cho provider (CLI override)

Không cần sửa `config.yaml`, có thể override trực tiếp bằng command:

```bash
# Dùng OpenAI ngay từ CLI
python main.py --provider openai --api-key "$OPENAI_API_KEY" --model gpt-5.4

# Dùng Gemini (mặc định)
python main.py --provider gemini --api-key "$GEMINI_API_KEY" --model gemini-2.0-flash

# Dùng LM Studio local
python main.py --provider lm_studio --api-base http://localhost:1234/v1 --api-key lm-studio --model qwen2.5-7b-instruct
```

`--provider` sẽ reset default theo provider đã chọn, sau đó áp dụng các override cụ thể (`--api-key`, `--model`, `--api-base`, `--temperature`, `--timeout`).

### Dump input trước khi gọi provider

Khi provider lỗi (429/quota/timeout), có thể dump toàn bộ input để tự request nơi khác:

```bash
python main.py --provider openai --dump-llm-input-dir ./llm_inputs --no-memory-store
```

Mỗi lần gọi LLM sẽ tạo một file `.md` chứa:
- `System Prompt`
- `User Content`

Bạn vẫn có thể gửi notification như bình thường, nhưng không lưu memory mới:

```bash
python main.py --provider openai --dump-llm-input-dir ./llm_inputs --no-memory-store --telegram
python main.py --provider openai --dump-llm-input-dir ./llm_inputs --no-memory-store --discord
python main.py --provider openai --dump-llm-input-dir ./llm_inputs --no-memory-store --discord --discord-attach-dump
```

### Chạy độc lập, không phụ thuộc provider LLM

Nếu chỉ cần xuất input để request ở hệ thống khác, dùng mode capture-only:

```bash
python main.py --dump-llm-input-only --dump-llm-input-dir ./llm_inputs --telegram
python main.py --dump-llm-input-only --dump-llm-input-dir ./llm_inputs --discord --discord-attach-dump
```

Mode này:
- Không gọi OpenAI/Gemini/LM Studio
- Tự động không lưu memory mới (tránh nhầm lẫn với nhận định chuẩn)
- Vẫn có thể gửi Telegram/Discord
- Nếu bật `--discord-attach-dump`, file `.md` dump sẽ được gửi kèm lên Discord

### Chọn kênh gửi notification

`--telegram` / `--discord` hoạt động ở cả 2 mode:

```bash
# Pipeline đầy đủ + chọn kênh gửi
python main.py --telegram                          # Chạy pipeline → gửi Telegram
python main.py --discord                           # Chạy pipeline → gửi Discord
python main.py                                     # Chạy pipeline → gửi tất cả (mặc định)

# Gửi lại báo cáo mới nhất (không chạy pipeline)
python main.py --notify-only                       # Gửi tất cả kênh đã bật
python main.py --notify-only --telegram            # Chỉ gửi Telegram
python main.py --notify-only --discord             # Chỉ gửi Discord
```

### Kết quả

```
reports/report_20260319.md
```

Cấu trúc báo cáo 6 mục:

```
## 1. 🛢️ Dầu & Kinh tế Vĩ mô
## 2. 🥇 Vàng & Kim loại quý
## 3. ₿ Crypto & Tài sản số
## 4. 🔄 Đối chiếu Xu hướng (so sánh với phiên trước)
## 5. ⚠️ Đánh giá Rủi ro & Cơ hội
## 6. 📋 Tổng kết
```

---

## 🔄 Feedback Loop

ChaHi **đối chiếu nhận định cũ** với tin mới mỗi phiên:

```
Phiên 1: Cào tin → LLM phân tích → Lưu nhận định vào Memory
                                         │
Phiên 2: Đọc nhận định cũ ←─────────────┘
         Cào tin mới → LLM đối chiếu → Lưu nhận định mới
                                         │
Phiên 3: ...                    ←────────┘
```

| Backend | Lưu ở đâu | Đặc điểm |
|---------|-----------|----------|
| **File** | `./memory/last_context.md` | 1 phiên gần nhất, ghi đè |
| **MCP** | ChromaDB (L1 per-workspace) | Nhiều phiên, tìm kiếm ngữ nghĩa |

---

## 🧠 MCP Server — Bộ nhớ dài hạn cho AI

ChaHi sử dụng **[TechStack Local MCP Server](https://github.com/PhanHug93/vibe-light-mcp)** làm Long-term Memory, cho phép AI đối chiếu nhận định qua nhiều phiên phân tích.

> 🧠 MCP server that gives AI coding agents **persistent memory**, **tech stack detection**, and **secure command execution**. Works with Cursor, VS Code, Claude, Windsurf & more.

### Tại sao ChaHi dùng MCP?

| Tính năng MCP | ChaHi sử dụng |
|---------------|---------------|
| 🧠 **Memory 2 tầng** (L1 + L2) | Lưu nhận định phiên → L1 per-workspace. Bài học kinh nghiệm → L2 global |
| 🔍 **Tìm kiếm ngữ nghĩa** | `search_memory` — tìm nhận định cũ liên quan bằng ChromaDB vector search |
| 🔄 **Auto-recall** | `auto_recall` — tự nhớ lại context phiên trước mà không cần query thủ công |
| 📦 **store_working_context** | Lưu tự động summary báo cáo sau mỗi phiên |
| 🛡️ **Chạy local, bảo mật** | Data không ra ngoài, ChromaDB chạy trên máy cá nhân |

### Quick Setup MCP cho ChaHi

```bash
# 1. Clone MCP Server
git clone https://github.com/PhanHug93/vibe-light-mcp.git
cd vibe-light-mcp

# 2. Cài đặt
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Chạy ChromaDB (Docker)
docker run -d --name chromadb -p 8888:8000 chromadb/chroma:latest

# 4. Chạy MCP Server
python main.py                        # stdio mode (mặc định)
python main.py --transport sse        # SSE mode (multi-client)
```

### Cấu hình ChaHi kết nối MCP

```yaml
# config.yaml
memory:
  type: "mcp"
  url: "http://localhost:8080"
  workspace_path: "/path/to/stock-light"
  protocol: "rest"                    # "rest" | "jsonrpc"
```

### MCP Tools mà ChaHi sử dụng

```
search_memory        → Tìm nhận định vĩ mô phiên gần nhất
auto_recall          → Fallback tự nhớ lại context
store_working_context → Lưu summary báo cáo sau mỗi phiên
```

> 📖 Xem đầy đủ tài liệu MCP Server: **[github.com/PhanHug93/vibe-light-mcp](https://github.com/PhanHug93/vibe-light-mcp)**

---

## 🧪 Phát triển & Test

```bash
python -m pytest tests/ -v              # Unit tests
python -m mypy src/chahi/ --strict      # Type checking
python -m ruff check src/ tests/        # Linting
```

### Cấu trúc thư mục

```
stock-light/
├── main.py                             # Entry point
├── config.example.yaml                 # Config mẫu
├── pyproject.toml                      # Dependencies & build
├── src/chahi/
│   ├── core/                           # Domain (không import bên ngoài)
│   │   ├── entities.py                 # DataClasses: Article, LLMSettings, NotificationSettings
│   │   ├── interfaces.py              # ABCs: INewsFetcher, ILLMClient, IMemoryManager, INotifier
│   │   └── use_cases.py               # Pipeline + SYSTEM_PROMPT + Feedback Loop
│   └── infrastructure/                 # I/O Layer
│       ├── config/
│       │   └── yaml_config_reader.py
│       ├── llm/
│       │   ├── lm_studio_client.py     # OpenAI-compatible
│       │   ├── gemini_client.py        # Google Gemini
│       │   ├── openai_client.py        # OpenAI API
│       │   └── llm_factory.py          # Strategy Pattern
│       ├── rss/
│       │   └── rss_fetcher.py          # RSS + BeautifulSoup
│       ├── memory/
│       │   ├── file_memory_manager.py  # File fallback
│       │   ├── mcp_memory_manager.py   # HTTP: REST + JSON-RPC SSE
│       │   └── memory_factory.py       # Factory dispatch
│       └── notifiers/
│           ├── telegram_notifier.py    # Telegram Bot API (auto-split 4096 chars)
│           ├── discord_notifier.py     # Discord Webhook (auto-split 2000 chars)
│           ├── notification_manager.py # Composite Pattern (fail-safe)
│           └── notifier_factory.py     # Factory dispatch
└── tests/unit/                         # 200+ tests
```

---

## ❓ Troubleshooting

| Vấn đề | Giải pháp |
|---------|-----------|
| `ConnectionError: Vui lòng kiểm tra LM Studio` | Mở LM Studio → Local Server → Start |
| `LLM timeout sau 120s` | Tăng `timeout` trong `config.yaml` (300-600) |
| Báo cáo bị cắt cụt | Tăng Context Length trong LM Studio lên 16384+ |
| RSS không có dữ liệu | Thay URL mới trong `config.yaml` |
| MCP connection refused | Kiểm tra server hoặc đổi `type: "file"` |
| MCP SSE handshake timeout | Kiểm tra server bật bridge mode, đúng port |
| Telegram không gửi được | Kiểm tra `chat_id` là **số** (vd: `-1001234567890`), thêm `@RawDataBot` vào group để lấy |
| Telegram Markdown lỗi | Tự động fallback plain text. Nếu vẫn lỗi, kiểm tra bot token |
| Discord webhook lỗi | Kiểm tra webhook URL chưa bị xóa trong Server Settings |

---

## 📄 License

MIT
