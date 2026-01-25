"""LLM client factory and management."""

import os
import logging
import streamlit as st
from llm.providers import OpenAIProvider, AnthropicProvider, GeminiProvider

logger = logging.getLogger(__name__)


def get_llm_provider():
    """Get configured LLM provider name."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if not provider:
        try:
            provider = st.secrets.get("LLM_PROVIDER", "openai").lower()
        except (KeyError, FileNotFoundError):
            provider = "openai"

    return provider


def get_api_key(provider):
    """Get API key for the specified provider."""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY"
    }

    key_name = key_map.get(provider)
    if not key_name:
        return None

    # Try environment variable first
    api_key = os.getenv(key_name)
    if api_key:
        return api_key

    # Fallback to streamlit secrets
    try:
        return st.secrets.get(key_name)
    except (KeyError, FileNotFoundError):
        return None


def create_llm_client():
    """Create and initialize LLM client based on configuration.

    Returns:
        tuple: (client, enabled, provider_name)
    """
    provider = get_llm_provider()
    api_key = get_api_key(provider)

    if not api_key:
        return None, False, provider

    try:
        if provider == "openai":
            client = OpenAIProvider(api_key)
        elif provider == "anthropic":
            client = AnthropicProvider(api_key)
        elif provider == "gemini":
            client = GeminiProvider(api_key)
        else:
            return None, False, provider

        return client, True, provider

    except Exception as e:
        logger.error(f"LLM init failed: {e}")
        return None, False, provider


def ask_agent(client, provider, ai_enabled, system_role, user_prompt, json_mode=False):
    """Call the LLM with the given prompt.

    Args:
        client: LLM client instance
        provider: Provider name (for error messages)
        ai_enabled: Whether AI is enabled
        system_role: System prompt
        user_prompt: User prompt
        json_mode: Whether to request JSON output

    Returns:
        str: LLM response or None if error
    """
    if not ai_enabled:
        st.warning(f"AI features disabled - check your {provider.upper()}_API_KEY in .env")
        return None

    try:
        with st.spinner("Processing..."):
            return client.call(system_role, user_prompt, json_mode)
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None
