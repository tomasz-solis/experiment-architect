"""Tests for LLM client helpers."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import streamlit as st

from llm.client import ask_agent_json, create_llm_client, get_llm_provider


class DummyClient:
    """Simple fake LLM client for retry tests."""

    def __init__(self, responses: list[str]) -> None:
        """Store queued responses for successive calls."""
        self._responses = responses
        self.name = "dummy"

    def call(self, system_role: str, user_prompt: str, json_mode: bool = False) -> str:
        """Return the next queued response."""
        return self._responses.pop(0)


@pytest.fixture
def patch_streamlit(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Patch Streamlit messaging helpers so tests can inspect them."""
    calls = {"warnings": [], "errors": []}

    monkeypatch.setattr(st, "spinner", lambda message: nullcontext())
    monkeypatch.setattr(st, "warning", lambda message: calls["warnings"].append(message))
    monkeypatch.setattr(st, "error", lambda message: calls["errors"].append(message))

    return calls


def test_ask_agent_json_retries_on_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
    patch_streamlit: dict[str, list[str]],
) -> None:
    """Malformed JSON should trigger one retry before succeeding."""
    monkeypatch.setattr("llm.client.time.sleep", lambda seconds: None)
    client = DummyClient(
        [
            "not valid json",
            '{"variant_col":"group","metric_col":"converted","metric_type":"binary"}',
        ]
    )

    result = ask_agent_json(
        client=client,
        provider="openai",
        ai_enabled=True,
        system_role="system",
        user_prompt="prompt",
        expected_keys=["variant_col", "metric_col", "metric_type"],
    )

    assert result is not None
    assert result["metric_type"] == "binary"
    assert patch_streamlit["errors"] == []


def test_ask_agent_json_returns_none_after_repeated_missing_keys(
    monkeypatch: pytest.MonkeyPatch,
    patch_streamlit: dict[str, list[str]],
) -> None:
    """Repeated incomplete JSON responses should surface a clear error."""
    monkeypatch.setattr("llm.client.time.sleep", lambda seconds: None)
    client = DummyClient(
        [
            '{"variant_col":"group"}',
            '{"variant_col":"group"}',
        ]
    )

    result = ask_agent_json(
        client=client,
        provider="openai",
        ai_enabled=True,
        system_role="system",
        user_prompt="prompt",
        expected_keys=["variant_col", "metric_col"],
    )

    assert result is None
    assert patch_streamlit["errors"]


def test_create_llm_client_disables_ai_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing credentials should disable the LLM layer cleanly."""
    monkeypatch.setattr("llm.client.get_llm_provider", lambda: "openai")
    monkeypatch.setattr("llm.client.check_optional_deps", lambda: {"anthropic": True, "gemini": True})
    monkeypatch.setattr("llm.client.get_api_key", lambda provider: None)

    client, enabled, provider = create_llm_client()

    assert client is None
    assert not enabled
    assert provider == "openai"


def test_get_llm_provider_defaults_to_openai_when_secret_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing-like secret value should still fall back to the default provider."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr("llm.client.st.secrets", {"LLM_PROVIDER": None})

    provider = get_llm_provider()

    assert provider == "openai"
