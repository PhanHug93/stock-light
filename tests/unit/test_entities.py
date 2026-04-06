"""Unit tests cho domain entities.

SourceCategory, SourceConfig, LLMSettings, Article, AnalysisContext.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from chahi.core.entities import (
    AnalysisContext,
    Article,
    LLMSettings,
    SourceCategory,
    SourceConfig,
)

# ─────────────────────────────────────────────────────────────
# SourceCategory
# ─────────────────────────────────────────────────────────────


class TestSourceCategory:
    """Test suite cho SourceCategory enum."""

    def test_has_three_members(self) -> None:
        """Enum phải có đúng 3 danh mục."""
        assert len(SourceCategory) == 3

    def test_values(self) -> None:
        """Kiểm tra giá trị string của từng member."""
        assert SourceCategory.OIL_MACRO.value == "oil_macro"
        assert SourceCategory.GOLD.value == "gold"
        assert SourceCategory.CRYPTO.value == "crypto"

    def test_str_display_labels(self) -> None:
        """__str__ phải trả về label hiển thị thân thiện."""
        assert str(SourceCategory.OIL_MACRO) == "Oil & Macro"
        assert str(SourceCategory.GOLD) == "Gold"
        assert str(SourceCategory.CRYPTO) == "Crypto"

    def test_from_value(self) -> None:
        """Có thể khởi tạo từ string value."""
        assert SourceCategory("oil_macro") is SourceCategory.OIL_MACRO
        assert SourceCategory("gold") is SourceCategory.GOLD

    def test_invalid_value_raises(self) -> None:
        """String không hợp lệ phải raise ValueError."""
        with pytest.raises(ValueError, match="'stocks'"):
            SourceCategory("stocks")


# ─────────────────────────────────────────────────────────────
# SourceConfig
# ─────────────────────────────────────────────────────────────


class TestSourceConfig:
    """Test suite cho SourceConfig."""

    def test_creation(self) -> None:
        """Khởi tạo thành công."""
        src = SourceConfig(name="Reuters", url="https://reuters.com/rss", type="rss")
        assert src.name == "Reuters"
        assert src.url == "https://reuters.com/rss"
        assert src.type == "rss"

    def test_default_type(self) -> None:
        """Type mặc định phải là 'rss'."""
        src = SourceConfig(name="Test", url="https://example.com")
        assert src.type == "rss"

    def test_frozen(self) -> None:
        """Không thể thay đổi attribute."""
        src = SourceConfig(name="Test", url="https://example.com")
        with pytest.raises(AttributeError):
            src.name = "Changed"  # type: ignore[misc]

    def test_empty_name_raises(self) -> None:
        """Name trống phải raise ValueError."""
        with pytest.raises(ValueError, match="name"):
            SourceConfig(name="  ", url="https://example.com")

    def test_empty_url_raises(self) -> None:
        """URL trống phải raise ValueError."""
        with pytest.raises(ValueError, match="url"):
            SourceConfig(name="Test", url="")

    def test_invalid_type_raises(self) -> None:
        """Type không hợp lệ phải raise ValueError."""
        with pytest.raises(ValueError, match="type"):
            SourceConfig(name="Test", url="https://example.com", type="api")


# ─────────────────────────────────────────────────────────────
# LLMSettings
# ─────────────────────────────────────────────────────────────


class TestLLMSettings:
    """Test suite cho LLMSettings."""

    def test_defaults(self) -> None:
        """Giá trị mặc định phải phù hợp Gemini."""
        llm = LLMSettings()
        assert llm.provider == "gemini"
        assert llm.api_base == ""
        assert llm.api_key == ""
        assert llm.model_name == "gemini-3.1-pro"
        assert llm.temperature == 0.1

    def test_custom_values(self) -> None:
        """Có thể override từng giá trị."""
        llm = LLMSettings(
            api_base="http://custom:8080/v1",
            api_key="custom-key",
            model_name="qwen2.5-7b",
            temperature=0.5,
        )
        assert llm.api_base == "http://custom:8080/v1"
        assert llm.model_name == "qwen2.5-7b"

    def test_openai_provider_valid(self) -> None:
        """Provider openai phải hợp lệ."""
        llm = LLMSettings(
            provider="openai",
            api_base="https://api.openai.com/v1",
            api_key="sk-test",
            model_name="gpt-5.4",
        )
        assert llm.provider == "openai"

    def test_frozen(self) -> None:
        """Không thể thay đổi attribute."""
        llm = LLMSettings()
        with pytest.raises(AttributeError):
            llm.temperature = 0.5  # type: ignore[misc]

    def test_empty_api_base_raises(self) -> None:
        """api_base trống phải raise ValueError khi dùng lm_studio."""
        with pytest.raises(ValueError, match="api_base"):
            LLMSettings(provider="lm_studio", api_base="")

    def test_temperature_too_high_raises(self) -> None:
        """Temperature > 2.0 phải raise ValueError."""
        with pytest.raises(ValueError, match="temperature"):
            LLMSettings(temperature=2.5)

    def test_temperature_negative_raises(self) -> None:
        """Temperature âm phải raise ValueError."""
        with pytest.raises(ValueError, match="temperature"):
            LLMSettings(temperature=-0.1)


# ─────────────────────────────────────────────────────────────
# Article
# ─────────────────────────────────────────────────────────────


class TestArticle:
    """Test suite cho Article."""

    @pytest.fixture()
    def sample_article(self) -> Article:
        """Tạo Article mẫu."""
        return Article(
            title="Fed giữ nguyên lãi suất",
            summary="Federal Reserve quyết định giữ lãi suất.",
            source_name="Reuters",
            published_date=datetime(2026, 3, 18, 10, 0, tzinfo=UTC),
            url="https://reuters.com/fed-rate",
        )

    def test_creation(self, sample_article: Article) -> None:
        """Khởi tạo thành công với đầy đủ fields."""
        assert sample_article.title == "Fed giữ nguyên lãi suất"
        assert sample_article.source_name == "Reuters"
        assert sample_article.published_date.year == 2026

    def test_frozen(self, sample_article: Article) -> None:
        """Không thể thay đổi attribute."""
        with pytest.raises(AttributeError):
            sample_article.title = "Changed"  # type: ignore[misc]

    def test_empty_title_raises(self) -> None:
        """Title trống phải raise ValueError."""
        with pytest.raises(ValueError, match="title"):
            Article(
                title="   ",
                summary="test",
                source_name="test",
                published_date=datetime.now(tz=UTC),
                url="https://example.com",
            )

    def test_empty_url_raises(self) -> None:
        """URL trống phải raise ValueError."""
        with pytest.raises(ValueError, match="url"):
            Article(
                title="Test",
                summary="test",
                source_name="test",
                published_date=datetime.now(tz=UTC),
                url="",
            )


# ─────────────────────────────────────────────────────────────
# AnalysisContext
# ─────────────────────────────────────────────────────────────


class TestAnalysisContext:
    """Test suite cho AnalysisContext."""

    @pytest.fixture()
    def sample_article(self) -> Article:
        """Tạo Article mẫu."""
        return Article(
            title="Test Article",
            summary="Summary",
            source_name="TestSource",
            published_date=datetime(2026, 3, 18, tzinfo=UTC),
            url="https://example.com/test",
        )

    def test_creation_empty(self) -> None:
        """Khởi tạo với danh sách rỗng."""
        ctx = AnalysisContext(date=date(2026, 3, 18))
        assert ctx.date == date(2026, 3, 18)
        assert ctx.oil_news == []
        assert ctx.gold_news == []
        assert ctx.crypto_news == []

    def test_total_articles(self, sample_article: Article) -> None:
        """total_articles phải tính tổng chính xác."""
        ctx = AnalysisContext(
            date=date(2026, 3, 18),
            oil_news=[sample_article, sample_article],
            gold_news=[sample_article],
            crypto_news=[],
        )
        assert ctx.total_articles == 3

    def test_is_empty_when_no_articles(self) -> None:
        """is_empty phải trả về True khi không có bài viết nào."""
        ctx = AnalysisContext(date=date(2026, 3, 18))
        assert ctx.is_empty is True

    def test_is_empty_when_has_articles(self, sample_article: Article) -> None:
        """is_empty phải trả về False khi có bài viết."""
        ctx = AnalysisContext(
            date=date(2026, 3, 18),
            oil_news=[sample_article],
        )
        assert ctx.is_empty is False

    def test_frozen(self, sample_article: Article) -> None:
        """Không thể reassign attribute (frozen)."""
        ctx = AnalysisContext(
            date=date(2026, 3, 18),
            oil_news=[sample_article],
        )
        with pytest.raises(AttributeError):
            ctx.date = date(2026, 1, 1)  # type: ignore[misc]
