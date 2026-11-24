"""Tests for the chunker module."""
from scripbox_kb.chunker import chunk_article, chunk_articles


def test_chunk_article_empty():
    assert chunk_article({}) == []
    assert chunk_article({"content": ""}) == []


def test_chunk_article_short():
    article = {
        "id": "art_1",
        "title": "Mutual Fund Basics",
        "url": "https://help.scripbox.com/mf",
        "category": "Mutual Funds",
        "folder": "General",
        "content": "This is a short article about mutual funds.",
    }
    chunks = chunk_article(article, chunk_words=50, overlap_words=10)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.article_id == "art_1"
    assert chunk.title == "Mutual Fund Basics"
    assert chunk.chunk_position == 0
    assert chunk.chunk_total == 1
    assert "mutual funds" in chunk.content


def test_chunk_article_overlap():
    words = [f"word{i}" for i in range(120)]
    article = {
        "id": "art_long",
        "title": "Long Guide",
        "content": " ".join(words),
    }
    chunks = chunk_article(article, chunk_words=50, overlap_words=10)
    assert len(chunks) >= 3
    # Verify overlap exists between chunk 0 and chunk 1
    c0_words = set(chunks[0].content.split())
    c1_words = set(chunks[1].content.split())
    overlap = c0_words.intersection(c1_words)
    assert len(overlap) == 10


def test_chunk_articles_multiple():
    articles = [
        {"id": "1", "title": "A", "content": "Hello world."},
        {"id": "2", "title": "B", "content": "Foo bar baz."},
    ]
    chunks = chunk_articles(articles)
    assert len(chunks) == 2
    assert chunks[0].article_id == "1"
    assert chunks[1].article_id == "2"
