"""Tests for the prompting module."""

from __future__ import annotations

from scripbox_kb.prompting import (
    STARTER_QUESTIONS,
    SYSTEM_PROMPT,
    WELCOME_MESSAGE,
    build_prompt,
    validate_input,
)


def test_system_prompt_exists() -> None:
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 0


def test_starter_questions_count() -> None:
    assert len(STARTER_QUESTIONS) == 6


def test_welcome_message_exists() -> None:
    assert isinstance(WELCOME_MESSAGE, str)
    assert len(WELCOME_MESSAGE) > 0


def test_build_prompt_basic(sample_hits: list[dict]) -> None:
    query = "How do I update my bank account?"
    prompt = build_prompt(query, sample_hits)
    assert "KNOWLEDGE BASE ARTICLES:" in prompt
    assert "USER QUESTION: How do I update my bank account?" in prompt
    assert "ANSWER:" in prompt
    assert sample_hits[0]["title"] in prompt
    assert sample_hits[1]["title"] in prompt


def test_build_prompt_truncation() -> None:
    long_doc = "A" * 5000
    hits = [
        {
            "title": "Long Doc",
            "url": "https://example.com",
            "category": "Cat",
            "folder": "Folder",
            "document": long_doc,
            "score": 0.9,
        }
    ]
    prompt = build_prompt("Query", hits)
    assert "A" * 1800 in prompt
    assert "A" * 1801 not in prompt


def test_build_prompt_custom_truncation() -> None:
    long_doc = "B" * 1000
    hits = [
        {
            "title": "Long Doc",
            "url": "https://example.com",
            "category": "Cat",
            "folder": "Folder",
            "document": long_doc,
            "score": 0.9,
        }
    ]
    prompt = build_prompt("Query", hits, max_doc_chars=500)
    assert "B" * 500 in prompt
    assert "B" * 501 not in prompt


def test_build_prompt_empty_hits() -> None:
    prompt = build_prompt("What?", [])
    assert prompt == "KNOWLEDGE BASE ARTICLES:\n\n\nUSER QUESTION: What?\n\nANSWER:"


def test_validate_input_valid() -> None:
    assert validate_input("What is KYC?") is None


def test_validate_input_empty() -> None:
    assert validate_input("") is not None
    assert validate_input("   ") is not None


def test_validate_input_too_long() -> None:
    long_query = "C" * 2001
    assert validate_input(long_query) is not None
    assert validate_input(long_query, max_length=2000) is not None
    assert validate_input("C" * 100, max_length=50) is not None
