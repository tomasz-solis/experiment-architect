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
        self.captured_prompts: list[str] = []
        self.name = "dummy"

    def call(self, system_role: str, user_prompt: str, json_mode: bool = False) -> str:
        """Return the next queued response and remember the prompt."""
        self.captured_prompts.append(user_prompt)
        return self._responses.pop(0)


@pytest.fixture
def patch_streamlit(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Patch Streamlit messaging helpers so tests can inspect them."""
    calls: dict[str, list[str]] = {"warnings": [], "errors": []}

    monkeypatch.setattr(st, "spinner", lambda message: nullcontext())
    monkeypatch.setattr(st, "warning", lambda message: calls["warnings"].append(message))
    monkeypatch.setattr(st, "error", lambda message: calls["errors"].append(message))

    return calls


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the retry backoff in every test."""
    monkeypatch.setattr("llm.client.time.sleep", lambda seconds: None)


def test_ask_agent_json_retries_on_malformed_json(
    patch_streamlit: dict[str, list[str]],
) -> None:
    """Malformed JSON should trigger one retry before succeeding."""
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
    patch_streamlit: dict[str, list[str]],
) -> None:
    """Repeated incomplete JSON responses should surface a clear error."""
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


def test_ask_agent_json_rejects_invalid_max_attempts(
    patch_streamlit: dict[str, list[str]],
) -> None:
    """max_attempts must be at least 1."""
    with pytest.raises(ValueError, match="at least 1"):
        ask_agent_json(
            client=DummyClient([]),
            provider="openai",
            ai_enabled=True,
            system_role="",
            user_prompt="",
            expected_keys=["foo"],
            max_attempts=0,
        )


def test_ask_agent_json_handles_non_dict_payload(
    patch_streamlit: dict[str, list[str]],
) -> None:
    """A JSON array (valid JSON but not a dict) should retry then fail."""
    client = DummyClient(['["not", "a", "dict"]', '["still", "not"]'])

    result = ask_agent_json(
        client=client,
        provider="openai",
        ai_enabled=True,
        system_role="role",
        user_prompt="prompt",
        expected_keys=["foo"],
        max_attempts=2,
    )

    assert result is None
    assert patch_streamlit["errors"]
    # The final error should mention that the payload was the wrong shape
    assert any("JSON object" in msg or "not a JSON" in msg for msg in patch_streamlit["errors"])


def test_ask_agent_json_retry_prompt_contains_expected_keys(
    patch_streamlit: dict[str, list[str]],
) -> None:
    """The stricter retry prompt must surface the expected keys to the model."""
    client = DummyClient(
        [
            "not json at all",
            '{"foo": "bar", "baz": "qux"}',
        ]
    )

    result = ask_agent_json(
        client=client,
        provider="openai",
        ai_enabled=True,
        system_role="role",
        user_prompt="original prompt",
        expected_keys=["foo", "baz"],
        max_attempts=2,
    )

    assert result == {"foo": "bar", "baz": "qux"}
    # The second prompt should reference the expected keys and the failure reason
    second_prompt = client.captured_prompts[1]
    assert "foo" in second_prompt
    assert "baz" in second_prompt
    assert "not valid JSON" in second_prompt


def test_ask_agent_json_retry_prompt_names_missing_keys(
    patch_streamlit: dict[str, list[str]],
) -> None:
    """When the retry is triggered by missing keys, the prompt should list them."""
    client = DummyClient(
        [
            '{"foo": "bar"}',  # missing baz
            '{"foo": "bar", "baz": "qux"}',
        ]
    )

    result = ask_agent_json(
        client=client,
        provider="openai",
        ai_enabled=True,
        system_role="role",
        user_prompt="original",
        expected_keys=["foo", "baz"],
        max_attempts=2,
    )

    assert result == {"foo": "bar", "baz": "qux"}
    second_prompt = client.captured_prompts[1]
    assert "baz" in second_prompt


def test_ask_agent_json_returns_none_when_ai_disabled(
    patch_streamlit: dict[str, list[str]],
) -> None:
    """If the AI layer is disabled, return None immediately without retrying."""
    client = DummyClient([])

    result = ask_agent_json(
        client=client,
        provider="openai",
        ai_enabled=False,
        system_role="role",
        user_prompt="prompt",
        expected_keys=["foo"],
    )

    assert result is None
    assert patch_streamlit["warnings"]


def test_create_llm_client_disables_ai_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing credentials should disable the LLM layer cleanly."""
    monkeypatch.setattr("llm.client.get_llm_provider", lambda: "openai")
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
