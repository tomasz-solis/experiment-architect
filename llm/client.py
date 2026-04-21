"""LLM client factory and request helpers."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import time
from typing import Any, Sequence

import streamlit as st

from llm.providers import AnthropicProvider, GeminiProvider, LLMProvider, OpenAIProvider

logger = logging.getLogger(__name__)

OPTIONAL_DEPS: dict[str, tuple[str, str]] = {
    "anthropic": ("anthropic", "pip install anthropic"),
    "gemini": ("google.generativeai", "pip install google-generativeai"),
    "statsmodels": ("statsmodels.api", "pip install statsmodels"),
}


def check_optional_deps() -> dict[str, bool]:
    """Report whether the optional runtime dependencies are installed."""
    availability: dict[str, bool] = {}
    for package_name, (import_path, _) in OPTIONAL_DEPS.items():
        availability[package_name] = importlib.util.find_spec(import_path) is not None

    return availability


def _normalize_provider_name(provider: str | None) -> str:
    """Return a normalized provider name with a safe default."""
    normalized = (provider or "").strip().lower()
    return normalized or "openai"


def get_llm_provider() -> str:
    """Return the configured LLM provider name."""
    provider = os.getenv("LLM_PROVIDER")

    if not provider:
        try:
            provider = st.secrets.get("LLM_PROVIDER")
        except (KeyError, FileNotFoundError):
            provider = "openai"

    return _normalize_provider_name(provider)


def get_api_key(provider: str) -> str | None:
    """Return the API key for the requested provider, if present."""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }

    key_name = key_map.get(provider)
    if not key_name:
        return None

    api_key = os.getenv(key_name)
    if api_key:
        return api_key

    try:
        return st.secrets.get(key_name)
    except (KeyError, FileNotFoundError):
        return None


def create_llm_client() -> tuple[LLMProvider | None, bool, str]:
    """Create the configured LLM client when credentials and deps are available."""
    provider = get_llm_provider()
    availability = check_optional_deps()

    if provider in ("anthropic", "gemini") and not availability.get(provider, False):
        install_hint = OPTIONAL_DEPS[provider][1]
        logger.warning("%s dependency missing. Install with: %s", provider, install_hint)
        return None, False, provider

    api_key = get_api_key(provider)
    if not api_key:
        logger.info("No API key configured for provider '%s'. AI features disabled.", provider)
        return None, False, provider

    try:
        if provider == "openai":
            client: LLMProvider = OpenAIProvider(api_key)
        elif provider == "anthropic":
            client = AnthropicProvider(api_key)
        elif provider == "gemini":
            client = GeminiProvider(api_key)
        else:
            logger.warning("Unsupported LLM provider configured: %s", provider)
            return None, False, provider

        logger.info("Initialized LLM provider '%s'.", provider)
        return client, True, provider
    except Exception as exc:
        logger.error("LLM init failed for provider '%s': %s", provider, exc)
        return None, False, provider


def _call_provider(
    client: LLMProvider | None,
    provider: str,
    ai_enabled: bool,
    system_role: str,
    user_prompt: str,
    json_mode: bool = False,
) -> str | None:
    """Call the configured provider and surface errors in the Streamlit UI."""
    if not ai_enabled or client is None:
        st.warning(f"AI features disabled - check your {provider.upper()}_API_KEY in .env")
        return None

    try:
        with st.spinner("Processing..."):
            return client.call(system_role, user_prompt, json_mode)
    except Exception as exc:
        logger.error("LLM request failed for provider '%s': %s", provider, exc)
        st.error(f"Error: {exc}")
        return None


def ask_agent(
    client: LLMProvider | None,
    provider: str,
    ai_enabled: bool,
    system_role: str,
    user_prompt: str,
    json_mode: bool = False,
) -> str | None:
    """Call the configured LLM and return its text response."""
    return _call_provider(client, provider, ai_enabled, system_role, user_prompt, json_mode)


def _build_retry_prompt(user_prompt: str, expected_keys: Sequence[str], reason: str) -> str:
    """Return a stricter retry prompt after a malformed JSON response."""
    keys = ", ".join(expected_keys)
    return (
        f"{user_prompt}\n\n"
        f"Your previous response could not be used because {reason}. "
        f"Return one valid JSON object with these keys only: {keys}."
    )


def ask_agent_json(
    client: LLMProvider | None,
    provider: str,
    ai_enabled: bool,
    system_role: str,
    user_prompt: str,
    expected_keys: Sequence[str],
    max_attempts: int = 2,
) -> dict[str, Any] | None:
    """Request JSON from the configured LLM with a small retry loop.

    Args:
        client: Initialized LLM provider adapter.
        provider: Provider name used for logging and UI messages.
        ai_enabled: Whether AI-assisted features are available.
        system_role: System instruction passed to the provider.
        user_prompt: User prompt passed to the provider.
        expected_keys: Keys that must be present in the JSON payload.
        max_attempts: Maximum number of attempts before surfacing an error.

    Returns:
        Parsed JSON object when the response is usable, otherwise ``None``.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    retry_prompt = user_prompt

    for attempt in range(1, max_attempts + 1):
        response = _call_provider(
            client=client,
            provider=provider,
            ai_enabled=ai_enabled,
            system_role=system_role,
            user_prompt=retry_prompt,
            json_mode=True,
        )
        if response is None:
            return None

        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Malformed JSON from provider '%s' on attempt %s/%s: %s",
                provider,
                attempt,
                max_attempts,
                exc,
            )
            if attempt == max_attempts:
                st.error("The model returned malformed JSON twice. Please try again.")
                return None
            retry_prompt = _build_retry_prompt(user_prompt, expected_keys, "it was not valid JSON")
            time.sleep(0.25 * attempt)
            continue

        if not isinstance(payload, dict):
            logger.warning(
                "Unexpected JSON payload type from provider '%s' on attempt %s/%s: %s",
                provider,
                attempt,
                max_attempts,
                type(payload).__name__,
            )
            if attempt == max_attempts:
                st.error("The model response was JSON, but not a JSON object. Please try again.")
                return None
            retry_prompt = _build_retry_prompt(
                user_prompt,
                expected_keys,
                f"it was a JSON {type(payload).__name__}, not an object",
            )
            time.sleep(0.25 * attempt)
            continue

        missing_keys = [key for key in expected_keys if key not in payload]
        if missing_keys:
            logger.warning(
                "Missing JSON keys from provider '%s' on attempt %s/%s: %s",
                provider,
                attempt,
                max_attempts,
                missing_keys,
            )
            if attempt == max_attempts:
                st.error(
                    "The model response was missing required keys: "
                    + ", ".join(missing_keys)
                    + "."
                )
                return None
            retry_prompt = _build_retry_prompt(
                user_prompt,
                expected_keys,
                f"it was missing these keys: {', '.join(missing_keys)}",
            )
            time.sleep(0.25 * attempt)
            continue

        logger.info(
            "Accepted JSON mapping from provider '%s' on attempt %s/%s.",
            provider,
            attempt,
            max_attempts,
        )
        return payload

    return None
