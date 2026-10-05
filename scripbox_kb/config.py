"""Configuration module for the Scripbox KB Retriever application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Application configuration settings."""

    chroma_dir: str = "./chroma_db"
    collection_name: str = "scripbox_kb"
    embed_model: str = "all-MiniLM-L6-v2"
    articles_file: str = "./articles.json"
    top_k: int = 5
    max_input_length: int = 2000
    max_context_chars: int = 1800
    max_output_tokens: int = 1024
    groq_api_key: str = ""
    gemini_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    gemini_model: str = "gemini-2.5-flash"

    @property
    def groq_ok(self) -> bool:
        """Check if GROQ API key is configured."""
        return bool(self.groq_api_key)

    @property
    def gemini_ok(self) -> bool:
        """Check if Gemini API key is configured."""
        return bool(self.gemini_api_key)

    @property
    def llm_ok(self) -> bool:
        """Check if at least one LLM API key is configured."""
        return self.groq_ok or self.gemini_ok


def resolve_key(secret_name: str) -> str:
    """
    Resolve a configuration key from Streamlit secrets or environment variables.

    Args:
        secret_name: The name of the secret/environment variable to resolve.

    Returns:
        The resolved key as a string, or an empty string if not found.
    """
    try:
        import streamlit as st

        key = st.secrets.get(secret_name, "")
        if key:
            return str(key)
    except Exception:
        pass

    return os.getenv(secret_name, "")


def load_config() -> AppConfig:
    """
    Load the application configuration.

    Returns:
        An initialized AppConfig instance with resolved API keys and models.
    """
    groq_model = resolve_key("GROQ_MODEL")
    gemini_model = resolve_key("GEMINI_MODEL")

    kwargs: dict[str, str] = {
        "groq_api_key": resolve_key("GROQ_API_KEY"),
        "gemini_api_key": resolve_key("GEMINI_API_KEY"),
    }

    if groq_model:
        kwargs["groq_model"] = groq_model

    if gemini_model:
        kwargs["gemini_model"] = gemini_model

    return AppConfig(**kwargs)  # type: ignore[arg-type]


def validate_config(config: AppConfig) -> list[str]:
    """
    Validate the application configuration.

    Args:
        config: The AppConfig instance to validate.

    Returns:
        A list of warning/error messages. Empty list if validation passes.
    """
    warnings: list[str] = []

    if not config.llm_ok:
        warnings.append("No LLM API keys configured. Set GROQ_API_KEY or GEMINI_API_KEY.")

    if not Path(config.chroma_dir).exists():
        warnings.append(f"Chroma directory '{config.chroma_dir}' does not exist.")

    if not Path(config.articles_file).exists():
        warnings.append(f"Articles file '{config.articles_file}' does not exist.")

    return warnings
