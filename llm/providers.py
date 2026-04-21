"""LLM provider implementations."""

from __future__ import annotations

from typing import Protocol

from config import (
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    MODEL_ANTHROPIC,
    MODEL_GEMINI,
    MODEL_OPENAI,
)


class LLMProvider(Protocol):
    """Common interface shared by all LLM provider adapters."""

    name: str

    def call(self, system_role: str, user_prompt: str, json_mode: bool = False) -> str:
        """Send a request and return the provider's text response."""


class OpenAIProvider:
    """OpenAI GPT provider."""

    def __init__(self, api_key: str) -> None:
        """Create an OpenAI client for the configured model."""
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.name = "openai"

    def call(self, system_role: str, user_prompt: str, json_mode: bool = False) -> str:
        """Send a request to OpenAI and return the response text."""
        response_format = {"type": "json_object"} if json_mode else None
        response = self.client.chat.completions.create(
            model=MODEL_OPENAI,
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": user_prompt},
            ],
            temperature=LLM_TEMPERATURE,
            response_format=response_format,
        )
        return response.choices[0].message.content


class AnthropicProvider:
    """Anthropic Claude provider."""

    def __init__(self, api_key: str) -> None:
        """Create an Anthropic client for the configured model."""
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.name = "anthropic"

    def call(self, system_role: str, user_prompt: str, json_mode: bool = False) -> str:
        """Send a request to Anthropic and return the response text."""
        prompt = f"{system_role}\n\n{user_prompt}"
        if json_mode:
            prompt += "\n\nRespond with valid JSON only."

        response = self.client.messages.create(
            model=MODEL_ANTHROPIC,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class GeminiProvider:
    """Google Gemini provider."""

    def __init__(self, api_key: str) -> None:
        """Create a Gemini client for the configured model."""
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.client = genai
        self.name = "gemini"

    def call(self, system_role: str, user_prompt: str, json_mode: bool = False) -> str:
        """Send a request to Gemini and return the response text."""
        model = self.client.GenerativeModel(MODEL_GEMINI)
        prompt = f"{system_role}\n\n{user_prompt}"
        if json_mode:
            prompt += "\n\nRespond with valid JSON only."

        response = model.generate_content(
            prompt,
            generation_config={"temperature": LLM_TEMPERATURE},
        )
        return response.text
