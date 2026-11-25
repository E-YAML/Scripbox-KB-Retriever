import pytest


@pytest.fixture
def sample_articles():
    """Returns a list of 3 sample article dictionaries for testing."""
    return [
        {
            "id": "1",
            "title": "How to invest in Mutual Funds",
            "url": "https://scripbox.com/mf/how-to-invest",
            "category": "Mutual Funds",
            "folder": "Investing",
            "meta_description": "A guide on how to invest in mutual funds.",
            "content": "Investing in mutual funds is simple...",
        },
        {
            "id": "2",
            "title": "What is ELSS?",
            "url": "https://scripbox.com/tax/what-is-elss",
            "category": "Tax Saving",
            "folder": "ELSS",
            "meta_description": "Learn about ELSS and tax saving.",
            "content": "ELSS stands for Equity Linked Savings Scheme...",
        },
        {
            "id": "3",
            "title": "Emergency Fund basics",
            "url": "https://scripbox.com/plan/emergency-fund",
            "category": "Financial Planning",
            "folder": "Emergency",
            "meta_description": "Why you need an emergency fund.",
            "content": "An emergency fund is crucial for financial stability...",
        },
    ]


@pytest.fixture
def sample_hits():
    """Returns a list of 2 retrieval hit dictionaries for testing."""
    return [
        {
            "title": "How to invest in Mutual Funds",
            "url": "https://scripbox.com/mf/how-to-invest",
            "category": "Mutual Funds",
            "folder": "Investing",
            "document": "Investing in mutual funds is simple...",
            "score": 0.95,
        },
        {
            "title": "What is ELSS?",
            "url": "https://scripbox.com/tax/what-is-elss",
            "category": "Tax Saving",
            "folder": "ELSS",
            "document": "ELSS stands for Equity Linked Savings Scheme...",
            "score": 0.82,
        },
    ]
