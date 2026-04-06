"""Unit tests cho LLM CLI override trong main.py."""

from __future__ import annotations

import argparse

from main import _apply_llm_cli_overrides

from chahi.core.entities import LLMSettings


def _make_args(
    *,
    provider: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        provider=provider,
        api_base=api_base,
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        timeout=timeout,
    )


class TestApplyLlmCliOverrides:
    """Tests cho _apply_llm_cli_overrides()."""

    def test_no_override_keeps_original(self) -> None:
        settings = LLMSettings(
            provider="gemini",
            api_key="g-key",
            model_name="gemini-2.0-flash",
            temperature=0.2,
            timeout=150,
        )
        args = _make_args()

        updated = _apply_llm_cli_overrides(settings, args)
        assert updated == settings

    def test_switch_to_openai_uses_provider_defaults(self) -> None:
        settings = LLMSettings(
            provider="gemini",
            api_key="g-key",
            model_name="gemini-2.0-flash",
        )
        args = _make_args(provider="openai")

        updated = _apply_llm_cli_overrides(settings, args)
        assert updated.provider == "openai"
        assert updated.api_base == "https://api.openai.com/v1"
        assert updated.api_key == ""
        assert updated.model_name == "gpt-5.4"

    def test_switch_to_openai_with_custom_overrides(self) -> None:
        settings = LLMSettings(
            provider="gemini",
            api_key="g-key",
            model_name="gemini-2.0-flash",
        )
        args = _make_args(
            provider="openai",
            api_key="sk-test",
            model_name="gpt-4.1-mini",
            temperature=0.3,
            timeout=300,
        )

        updated = _apply_llm_cli_overrides(settings, args)
        assert updated.provider == "openai"
        assert updated.api_base == "https://api.openai.com/v1"
        assert updated.api_key == "sk-test"
        assert updated.model_name == "gpt-4.1-mini"
        assert updated.temperature == 0.3
        assert updated.timeout == 300

    def test_switch_to_lm_studio_sets_local_defaults(self) -> None:
        settings = LLMSettings(
            provider="gemini",
            api_key="g-key",
            model_name="gemini-2.0-flash",
        )
        args = _make_args(provider="lm_studio")

        updated = _apply_llm_cli_overrides(settings, args)
        assert updated.provider == "lm_studio"
        assert updated.api_base == "http://localhost:1234/v1"
        assert updated.api_key == "lm-studio"
        assert updated.model_name == "default"
