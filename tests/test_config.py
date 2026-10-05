"""Tests for the configuration module."""

from scripbox_kb.config import AppConfig, resolve_key, validate_config


def test_default_config():
    """Test AppConfig with default values."""
    config = AppConfig()
    assert config.chroma_dir == "./chroma_db"
    assert config.collection_name == "scripbox_kb"
    assert config.embed_model == "all-MiniLM-L6-v2"
    assert config.articles_file == "./articles.json"
    assert config.top_k == 5
    assert config.max_input_length == 2000
    assert config.max_context_chars == 1800
    assert config.max_output_tokens == 1024
    assert config.groq_api_key == ""
    assert config.gemini_api_key == ""
    assert config.groq_model == "openai/gpt-oss-120b"
    assert config.gemini_model == "gemini-2.5-flash"


def test_groq_ok_with_key():
    """Test groq_ok property with a configured key."""
    config = AppConfig(groq_api_key="sk-groq...")
    assert config.groq_ok is True


def test_groq_ok_without_key():
    """Test groq_ok property without a configured key."""
    config = AppConfig()
    assert config.groq_ok is False


def test_llm_ok_either_key():
    """Test llm_ok property with either key configured."""
    config_groq = AppConfig(groq_api_key="sk-groq...")
    assert config_groq.llm_ok is True

    config_gemini = AppConfig(gemini_api_key="AIza...")
    assert config_gemini.llm_ok is True

    config_both = AppConfig(groq_api_key="sk-groq...", gemini_api_key="AIza...")
    assert config_both.llm_ok is True


def test_llm_ok_no_keys():
    """Test llm_ok property with no keys configured."""
    config = AppConfig()
    assert config.llm_ok is False


def test_validate_config_no_keys():
    """Test validation with no keys configured."""
    config = AppConfig()
    warnings = validate_config(config)
    assert any("No LLM API keys configured" in w for w in warnings)


def test_resolve_key_from_env(monkeypatch):
    """Test resolve_key with environment variables."""
    monkeypatch.setenv("TEST_DUMMY_KEY", "dummy_value")
    assert resolve_key("TEST_DUMMY_KEY") == "dummy_value"


def test_resolve_key_missing():
    """Test resolve_key with a missing key."""
    assert resolve_key("MISSING_KEY_XYZ") == ""
