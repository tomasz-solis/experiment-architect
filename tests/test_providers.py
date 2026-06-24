"""Tests for the LLM provider adapters using stubbed SDK clients.

The provider ``call()`` methods own the per-vendor request shaping (message
roles, JSON-mode handling, response extraction). These tests stub each vendor
SDK so that logic is exercised without network access or the optional
``anthropic`` / ``google-generativeai`` packages being installed.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from config import MODEL_ANTHROPIC, MODEL_OPENAI
from llm.providers import OpenAIProvider


class TestOpenAIProvider:
    """OpenAIProvider.call shapes a chat-completions request."""

    def _provider(self, content: str | None) -> tuple[OpenAIProvider, MagicMock]:
        provider = OpenAIProvider("sk-test")  # constructor does no network I/O
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = content
        client.chat.completions.create.return_value.choices = [choice]
        provider.client = client
        return provider, client

    def test_returns_message_content(self) -> None:
        provider, _ = self._provider("mapped json")
        assert provider.call("system", "user") == "mapped json"

    def test_none_content_becomes_empty_string(self) -> None:
        provider, _ = self._provider(None)
        assert provider.call("system", "user") == ""

    def test_json_mode_sets_response_format_and_roles(self) -> None:
        provider, client = self._provider("{}")
        provider.call("be terse", "map these", json_mode=True)

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == MODEL_OPENAI
        assert kwargs["response_format"] == {"type": "json_object"}
        assert [m["role"] for m in kwargs["messages"]] == ["system", "user"]
        assert kwargs["messages"][0]["content"] == "be terse"
        assert kwargs["messages"][1]["content"] == "map these"


@pytest.fixture
def stub_anthropic(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a fake ``anthropic`` module and return the stub message client."""
    messages_client = MagicMock()
    block = MagicMock()
    block.text = "anthropic reply"
    messages_client.create.return_value.content = [block]

    fake_client = MagicMock()
    fake_client.messages = messages_client
    fake_module = types.ModuleType("anthropic")
    fake_module.Anthropic = MagicMock(return_value=fake_client)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    return messages_client


@pytest.fixture
def stub_gemini(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a fake ``google.generativeai`` module and return the stub model."""
    model = MagicMock()
    model.generate_content.return_value.text = "gemini reply"

    fake_genai = types.ModuleType("google.generativeai")
    fake_genai.configure = MagicMock()  # type: ignore[attr-defined]
    fake_genai.GenerativeModel = MagicMock(return_value=model)  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", types.ModuleType("google"))
    monkeypatch.setitem(sys.modules, "google.generativeai", fake_genai)
    return model


class TestAnthropicProvider:
    """AnthropicProvider folds system+user into one prompt and reads content[0]."""

    def test_returns_first_text_block(self, stub_anthropic: MagicMock) -> None:
        from llm.providers import AnthropicProvider

        provider = AnthropicProvider("key")
        assert provider.call("system", "user") == "anthropic reply"

    def test_json_mode_appends_instruction(self, stub_anthropic: MagicMock) -> None:
        from llm.providers import AnthropicProvider

        AnthropicProvider("key").call("system", "user", json_mode=True)
        prompt = stub_anthropic.create.call_args.kwargs["messages"][0]["content"]
        assert prompt.startswith("system\n\nuser")
        assert "Respond with valid JSON only." in prompt
        assert stub_anthropic.create.call_args.kwargs["model"] == MODEL_ANTHROPIC


class TestGeminiProvider:
    """GeminiProvider configures the SDK and reads response.text."""

    def test_returns_response_text(self, stub_gemini: MagicMock) -> None:
        from llm.providers import GeminiProvider

        provider = GeminiProvider("key")
        assert provider.call("system", "user") == "gemini reply"

    def test_json_mode_appends_instruction(self, stub_gemini: MagicMock) -> None:
        from llm.providers import GeminiProvider

        GeminiProvider("key").call("system", "user", json_mode=True)
        prompt = stub_gemini.generate_content.call_args.args[0]
        assert "Respond with valid JSON only." in prompt
