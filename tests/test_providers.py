from __future__ import annotations

from unittest.mock import patch

import pytest

from scripbox_kb.providers import (
    ProviderAuthError,
    ProviderModelNotFoundError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    classify_error,
    stream_with_fallback,
)


def test_classify_rate_limit():
    exc = Exception("HTTP 429 Too Many Requests")
    res = classify_error(exc, "test")
    assert isinstance(res, ProviderRateLimitError)


def test_classify_rate_limit_text():
    exc = Exception("rate_limit_exceeded")
    res = classify_error(exc, "test")
    assert isinstance(res, ProviderRateLimitError)


def test_classify_auth():
    exc = Exception("401 Unauthorized")
    res = classify_error(exc, "test")
    assert isinstance(res, ProviderAuthError)


def test_classify_auth_text():
    exc = Exception("invalid_api_key")
    res = classify_error(exc, "test")
    assert isinstance(res, ProviderAuthError)


def test_classify_model_not_found():
    exc = Exception("404 Model Not Found")
    res = classify_error(exc, "test")
    assert isinstance(res, ProviderModelNotFoundError)


def test_classify_decommissioned():
    exc = Exception("model decommissioned")
    res = classify_error(exc, "test")
    assert isinstance(res, ProviderModelNotFoundError)


def test_classify_generic():
    exc = Exception("connection timeout")
    res = classify_error(exc, "test")
    assert isinstance(res, ProviderUnavailableError)


def test_classify_preserves_original():
    orig = Exception("original")
    res = classify_error(orig, "test")
    assert res.original is orig


def test_classify_preserves_provider():
    res = classify_error(Exception("err"), "my_provider")
    assert res.provider == "my_provider"


@patch("scripbox_kb.providers.groq_stream")
@patch("scripbox_kb.providers.gemini_stream")
def test_fallback_groq_rate_limit_to_gemini(mock_gemini, mock_groq):
    def mock_groq_gen(*args, **kwargs):
        raise ProviderRateLimitError("rate limit", "groq")
        yield

    mock_groq.side_effect = mock_groq_gen

    def mock_gemini_gen(*args, **kwargs):
        yield "hello"

    mock_gemini.side_effect = mock_gemini_gen

    result = stream_with_fallback(
        "prompt",
        groq_api_key="groq-key",
        gemini_api_key="gemini-key",
        groq_model="groq-model",
        gemini_model="gemini-model",
        system_prompt="sys",
    )

    assert result.provider == "gemini"
    assert result.fallback_reason == "Groq fallback: ProviderRateLimitError"
    assert list(result.stream) == ["hello"]


@patch("scripbox_kb.providers.groq_stream")
def test_fallback_groq_auth_raises(mock_groq):
    def mock_groq_gen(*args, **kwargs):
        raise ProviderAuthError("auth", "groq")
        yield

    mock_groq.side_effect = mock_groq_gen

    with pytest.raises(ProviderAuthError):
        stream_with_fallback(
            "prompt",
            groq_api_key="groq-key",
            gemini_api_key="gemini-key",
            groq_model="groq-model",
            gemini_model="gemini-model",
            system_prompt="sys",
        )


def test_no_providers_raises():
    with pytest.raises(ProviderUnavailableError):
        stream_with_fallback(
            "prompt",
            groq_api_key="",
            gemini_api_key="",
            groq_model="groq-model",
            gemini_model="gemini-model",
            system_prompt="sys",
        )
