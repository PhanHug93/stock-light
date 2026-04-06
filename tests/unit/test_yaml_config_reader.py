"""Unit tests cho YamlConfigReader."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest
import yaml

from chahi.core.entities import LLMSettings, SourceCategory, SourceConfig
from chahi.infrastructure.config.yaml_config_reader import YamlConfigReader

if TYPE_CHECKING:
    from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Test Config Strings
# ─────────────────────────────────────────────────────────────

_VALID_CONFIG = dedent("""\
    sources:
      oil_macro:
        - name: "Reuters Business"
          url: "https://feeds.reuters.com/reuters/businessNews"
          type: rss
        - name: "Reuters Energy"
          url: "https://feeds.reuters.com/reuters/UKEnergyNews"
          type: rss
      gold:
        - name: "Kitco Gold"
          url: "https://www.kitco.com/rss/gold.xml"
          type: rss
      crypto:
        - name: "CoinDesk"
          url: "https://www.coindesk.com/arc/outboundfeeds/rss/"
          type: rss

    llm_settings:
      api_base: "http://localhost:5555/v1"
      api_key: "test-key"
      model_name: "qwen2.5-7b-instruct"
      temperature: 0.2
""")

_MINIMAL_CONFIG = dedent("""\
    sources:
      gold:
        - name: "Kitco"
          url: "https://www.kitco.com/rss/gold.xml"
          type: rss
""")

_OPENAI_CONFIG = dedent("""\
    sources:
      gold:
        - name: "Kitco"
          url: "https://www.kitco.com/rss/gold.xml"
          type: rss

    llm_settings:
      provider: "openai"
""")

_NO_SOURCES_CONFIG = dedent("""\
    llm_settings:
      api_base: "http://localhost:1234/v1"
""")

_INVALID_CATEGORY = dedent("""\
    sources:
      forex:
        - name: "Forex Feed"
          url: "https://example.com/forex"
          type: rss
""")

_MISSING_URL = dedent("""\
    sources:
      gold:
        - name: "Bad Feed"
          type: rss
""")

_MISSING_NAME = dedent("""\
    sources:
      crypto:
        - url: "https://example.com/feed"
          type: rss
""")

_EMPTY_CATEGORY = dedent("""\
    sources:
      gold: []
""")

_BAD_YAML = "sources: [\ninvalid: {{"


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────


def _write_config(tmp_path: Path, content: str) -> Path:
    """Helper: ghi content vào file config.yaml tạm."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(content, encoding="utf-8")
    return config_file


@pytest.fixture()
def valid_config(tmp_path: Path) -> Path:
    """File config đầy đủ."""
    return _write_config(tmp_path, _VALID_CONFIG)


@pytest.fixture()
def minimal_config(tmp_path: Path) -> Path:
    """File config tối thiểu (chỉ sources, không llm_settings)."""
    return _write_config(tmp_path, _MINIMAL_CONFIG)


# ─────────────────────────────────────────────────────────────
# get_sources() — Happy Path
# ─────────────────────────────────────────────────────────────


class TestGetSourcesValid:
    """Tests cho get_sources() với config hợp lệ."""

    def test_returns_three_categories(self, valid_config: Path) -> None:
        """Config đầy đủ phải trả về 3 categories."""
        reader = YamlConfigReader(config_path=valid_config)
        sources = reader.get_sources()
        assert len(sources) == 3

    def test_oil_macro_has_two_sources(self, valid_config: Path) -> None:
        """oil_macro phải có 2 sources."""
        reader = YamlConfigReader(config_path=valid_config)
        sources = reader.get_sources()
        assert len(sources[SourceCategory.OIL_MACRO]) == 2

    def test_source_config_fields(self, valid_config: Path) -> None:
        """SourceConfig phải chứa đúng name, url, type."""
        reader = YamlConfigReader(config_path=valid_config)
        sources = reader.get_sources()
        first = sources[SourceCategory.OIL_MACRO][0]

        assert isinstance(first, SourceConfig)
        assert first.name == "Reuters Business"
        assert "reuters" in first.url
        assert first.type == "rss"

    def test_gold_category_key(self, valid_config: Path) -> None:
        """Category GOLD phải tồn tại với key đúng."""
        reader = YamlConfigReader(config_path=valid_config)
        sources = reader.get_sources()
        assert SourceCategory.GOLD in sources
        assert sources[SourceCategory.GOLD][0].name == "Kitco Gold"

    def test_minimal_config_one_category(self, minimal_config: Path) -> None:
        """Config tối thiểu chỉ có 1 category."""
        reader = YamlConfigReader(config_path=minimal_config)
        sources = reader.get_sources()
        assert len(sources) == 1
        assert SourceCategory.GOLD in sources


# ─────────────────────────────────────────────────────────────
# get_sources() — Error Cases
# ─────────────────────────────────────────────────────────────


class TestGetSourcesErrors:
    """Tests cho get_sources() với config không hợp lệ."""

    def test_file_not_found(self, tmp_path: Path) -> None:
        """File không tồn tại phải raise FileNotFoundError."""
        reader = YamlConfigReader(config_path=tmp_path / "ghost.yaml")
        with pytest.raises(FileNotFoundError, match="không tồn tại"):
            reader.get_sources()

    def test_empty_file(self, tmp_path: Path) -> None:
        """File rỗng phải raise ValueError."""
        config = _write_config(tmp_path, "")
        reader = YamlConfigReader(config_path=config)
        with pytest.raises(ValueError, match="dictionary"):
            reader.get_sources()

    def test_no_sources_key(self, tmp_path: Path) -> None:
        """Config thiếu 'sources' phải raise ValueError."""
        config = _write_config(tmp_path, _NO_SOURCES_CONFIG)
        reader = YamlConfigReader(config_path=config)
        with pytest.raises(ValueError, match="sources"):
            reader.get_sources()

    def test_invalid_category(self, tmp_path: Path) -> None:
        """Category không hợp lệ phải raise ValueError."""
        config = _write_config(tmp_path, _INVALID_CATEGORY)
        reader = YamlConfigReader(config_path=config)
        with pytest.raises(ValueError, match="forex"):
            reader.get_sources()

    def test_source_missing_url(self, tmp_path: Path) -> None:
        """Source thiếu URL phải raise ValueError."""
        config = _write_config(tmp_path, _MISSING_URL)
        reader = YamlConfigReader(config_path=config)
        with pytest.raises(ValueError, match="url"):
            reader.get_sources()

    def test_source_missing_name(self, tmp_path: Path) -> None:
        """Source thiếu name phải raise ValueError."""
        config = _write_config(tmp_path, _MISSING_NAME)
        reader = YamlConfigReader(config_path=config)
        with pytest.raises(ValueError, match="name"):
            reader.get_sources()

    def test_empty_category_list(self, tmp_path: Path) -> None:
        """Category rỗng phải raise ValueError."""
        config = _write_config(tmp_path, _EMPTY_CATEGORY)
        reader = YamlConfigReader(config_path=config)
        with pytest.raises(ValueError, match="rỗng"):
            reader.get_sources()

    def test_bad_yaml_syntax(self, tmp_path: Path) -> None:
        """YAML lỗi cú pháp phải raise exception."""
        config = _write_config(tmp_path, _BAD_YAML)
        reader = YamlConfigReader(config_path=config)
        with pytest.raises((ValueError, yaml.YAMLError)):
            reader.get_sources()

    def test_yaml_not_dict(self, tmp_path: Path) -> None:
        """YAML là list thay vì dict phải raise ValueError."""
        config = _write_config(tmp_path, "- item1\n- item2\n")
        reader = YamlConfigReader(config_path=config)
        with pytest.raises(ValueError, match="dictionary"):
            reader.get_sources()


# ─────────────────────────────────────────────────────────────
# get_llm_settings()
# ─────────────────────────────────────────────────────────────


class TestGetLLMSettings:
    """Tests cho get_llm_settings()."""

    def test_custom_values(self, valid_config: Path) -> None:
        """LLM settings phải parse đúng giá trị custom."""
        reader = YamlConfigReader(config_path=valid_config)
        llm = reader.get_llm_settings()

        assert isinstance(llm, LLMSettings)
        assert llm.api_base == "http://localhost:5555/v1"
        assert llm.api_key == "test-key"
        assert llm.model_name == "qwen2.5-7b-instruct"
        assert llm.temperature == 0.2

    def test_defaults_when_missing(self, minimal_config: Path) -> None:
        """Khi thiếu llm_settings, phải dùng giá trị mặc định."""
        reader = YamlConfigReader(config_path=minimal_config)
        llm = reader.get_llm_settings()

        assert llm.provider == "gemini"
        assert llm.api_base == ""
        assert llm.api_key == ""
        assert llm.model_name == "gemini-2.0-flash"
        assert llm.temperature == 0.1

    def test_openai_provider_defaults(self, tmp_path: Path) -> None:
        """provider=openai phải dùng defaults đúng cho OpenAI."""
        config = _write_config(tmp_path, _OPENAI_CONFIG)
        reader = YamlConfigReader(config_path=config)
        llm = reader.get_llm_settings()

        assert llm.provider == "openai"
        assert llm.api_base == "https://api.openai.com/v1"
        assert llm.api_key == ""
        assert llm.model_name == "gpt-4o-mini"

    def test_provider_is_case_insensitive(self, tmp_path: Path) -> None:
        """Provider viết hoa/thường lẫn nhau vẫn parse đúng."""
        config = _write_config(
            tmp_path,
            dedent("""\
                sources:
                  gold:
                    - name: "Kitco"
                      url: "https://www.kitco.com/rss/gold.xml"
                      type: rss
                llm_settings:
                  provider: "OpenAI"
            """),
        )
        reader = YamlConfigReader(config_path=config)
        llm = reader.get_llm_settings()
        assert llm.provider == "openai"

    def test_immutable(self, valid_config: Path) -> None:
        """LLMSettings phải immutable."""
        reader = YamlConfigReader(config_path=valid_config)
        llm = reader.get_llm_settings()
        with pytest.raises(AttributeError):
            llm.api_base = "changed"  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────
# Caching behavior
# ─────────────────────────────────────────────────────────────


class TestCaching:
    """Tests cho lazy loading + cache."""

    def test_consistent_results(self, valid_config: Path) -> None:
        """Gọi get_sources() nhiều lần phải trả về kết quả nhất quán."""
        reader = YamlConfigReader(config_path=valid_config)
        first = reader.get_sources()
        second = reader.get_sources()
        assert first.keys() == second.keys()

    def test_mixed_calls(self, valid_config: Path) -> None:
        """Gọi xen kẽ get_sources() và get_llm_settings() phải hoạt động."""
        reader = YamlConfigReader(config_path=valid_config)
        sources = reader.get_sources()
        llm = reader.get_llm_settings()
        sources_again = reader.get_sources()

        assert len(sources) == len(sources_again)
        assert llm.model_name == "qwen2.5-7b-instruct"
