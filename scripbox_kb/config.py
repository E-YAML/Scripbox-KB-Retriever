"""Configuration module for the Scripbox KB Retriever application."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """Application configuration settings."""

    # ── Paths ──────────────────────────────────────────────────────────────────
    chroma_dir: str = "./chroma_db"
    collection_name: str = "scripbox_kb"
    articles_file: str = "./articles.json"

    # ── Embedding model ────────────────────────────────────────────────────────
    embed_model: str = "all-MiniLM-L6-v2"

    # ── Retrieval settings ─────────────────────────────────────────────────────
    top_k: int = 5  # final sources shown / passed to LLM
    retrieval_candidates: int = 20  # over-fetch before dedup + threshold filter
    retrieval_threshold: float = 0.35  # min cosine similarity to pass grading

    # ── LLM limits ─────────────────────────────────────────────────────────────
    max_input_length: int = 2000
    max_context_chars: int = 1800
    max_output_tokens: int = 1024

    # ── LLM provider keys + models ─────────────────────────────────────────────
    groq_api_key: str = ""
    gemini_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    gemini_model: str = "gemini-2.5-flash"

    # ── Vector DB backend ──────────────────────────────────────────────────────
    vector_db_type: str = "chroma"  # "chroma" | "qdrant"
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # ── Graph settings ─────────────────────────────────────────────────────────
    max_rewrites: int = 2  # max query-rewrite cycles before abstaining

    # ── Reliability & Concurrency ──────────────────────────────────────────────
    max_concurrency: int = 4  # max simultaneous active queries
    concurrency_timeout: float = 15.0  # seconds to wait for a slot before returning busy

    # ── Properties ─────────────────────────────────────────────────────────────

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

    @property
    def qdrant_ok(self) -> bool:
        """Check if Qdrant is properly configured."""
        return self.vector_db_type == "qdrant" and bool(self.qdrant_url)


def resolve_key(secret_name: str) -> str:
    """Resolve a config key from Streamlit secrets, secrets.toml, or environment variables.

    Args:
        secret_name: The name of the secret/environment variable.

    Returns:
        The resolved value as a string, or empty string if not found.
    """
    try:
        import streamlit as st

        key = st.secrets.get(secret_name, "")
        if key:
            return str(key)
    except Exception:
        pass

    # Fallback to direct reading of .streamlit/secrets.toml
    try:
        import tomllib

        secrets_path = Path(".streamlit/secrets.toml")
        if secrets_path.exists():
            with open(secrets_path, "rb") as f:
                data = tomllib.load(f)
                val = data.get(secret_name, "")
                if val:
                    return str(val)
    except Exception:
        pass

    return os.getenv(secret_name, "")


def load_config() -> AppConfig:
    """Load the application configuration.

    Returns:
        An initialized AppConfig instance with resolved API keys and models.
    """
    groq_model = resolve_key("GROQ_MODEL")
    gemini_model = resolve_key("GEMINI_MODEL")
    vector_db_type = resolve_key("VECTOR_DB_TYPE") or "chroma"
    qdrant_url = resolve_key("QDRANT_URL")
    qdrant_api_key = resolve_key("QDRANT_API_KEY")

    kwargs: dict[str, str | int | float] = {
        "groq_api_key": resolve_key("GROQ_API_KEY"),
        "gemini_api_key": resolve_key("GEMINI_API_KEY"),
        "vector_db_type": vector_db_type,
        "qdrant_url": qdrant_url,
        "qdrant_api_key": qdrant_api_key,
    }

    if groq_model:
        kwargs["groq_model"] = groq_model
    if gemini_model:
        kwargs["gemini_model"] = gemini_model

    # Optional numeric overrides from env
    top_k_raw = resolve_key("TOP_K")
    threshold_raw = resolve_key("RETRIEVAL_THRESHOLD")
    if top_k_raw:
        try:
            kwargs["top_k"] = int(top_k_raw)
        except ValueError:
            pass
    if threshold_raw:
        try:
            kwargs["retrieval_threshold"] = float(threshold_raw)
        except ValueError:
            pass

    # Optional release pointer from data/releases/current.json
    current_rel_file = Path("data/releases/current.json")
    if current_rel_file.exists():
        try:
            with open(current_rel_file, encoding="utf-8") as f:
                rel_meta = json.load(f)
            if "collection_name" in rel_meta and not resolve_key("COLLECTION_NAME"):
                kwargs["collection_name"] = rel_meta["collection_name"]
            if (
                "chroma_dir" in rel_meta
                and rel_meta["chroma_dir"]
                and not resolve_key("CHROMA_DIR")
            ):
                kwargs["chroma_dir"] = rel_meta["chroma_dir"]
            if "articles_file" in rel_meta and not resolve_key("ARTICLES_FILE"):
                kwargs["articles_file"] = rel_meta["articles_file"]
            if "backend" in rel_meta and not resolve_key("VECTOR_DB_TYPE"):
                kwargs["vector_db_type"] = rel_meta["backend"]
        except Exception:
            pass

    return AppConfig(**kwargs)  # type: ignore[arg-type]


def validate_config(config: AppConfig) -> list[str]:
    """Validate the application configuration.

    Args:
        config: The AppConfig instance to validate.

    Returns:
        A list of warning/error messages. Empty list means config is valid.
    """
    warnings: list[str] = []

    if not config.llm_ok:
        warnings.append("No LLM API keys configured. Set GROQ_API_KEY or GEMINI_API_KEY.")

    if config.vector_db_type == "qdrant":
        if not config.qdrant_url:
            warnings.append("VECTOR_DB_TYPE=qdrant but QDRANT_URL is not set.")
    else:
        if not Path(config.chroma_dir).exists():
            warnings.append(f"Chroma directory '{config.chroma_dir}' does not exist.")

    if not Path(config.articles_file).exists():
        warnings.append(f"Articles file '{config.articles_file}' does not exist.")

    return warnings
