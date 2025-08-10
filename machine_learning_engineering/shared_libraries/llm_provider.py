"""LLM provider abstraction and OpenAI-compatible implementation.

This module provides a minimal interface to call an OpenAI-compatible API
(OpenRouter, LM Studio, Ollama)."""

from __future__ import annotations

import os
from typing import List, Dict


class OpenAICompatProvider:
    """OpenAI-compatible provider (OpenRouter/Ollama/LM Studio)."""

    def __init__(self) -> None:
        # Import inside to avoid import side-effects during static analysis
        from openai import OpenAI  # type: ignore

        self._base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        self._api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("LLM_MODEL") or os.getenv(
            "ROOT_AGENT_MODEL", "meta-llama/llama-3.1-8b-instruct:free"
        )
        self.referer = os.getenv("OPENROUTER_SITE_URL", "http://localhost")
        self.app_name = os.getenv("OPENROUTER_APP_NAME", "machine-learning-engineering")

        self.client = OpenAI(base_url=self._base_url, api_key=self._api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        # Add OpenRouter-specific headers only when targeting OpenRouter
        extra_headers = None
        if "openrouter.ai" in (self._base_url or ""):
            extra_headers = {"HTTP-Referer": self.referer, "X-Title": self.app_name}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
        )
        content = resp.choices[0].message.content
        return content or ""


def get_llm() -> OpenAICompatProvider:
    return OpenAICompatProvider()
