"""Semantic text chunker for Scripbox KB articles.

Splits articles into overlapping word-based chunks, retaining article metadata
on every chunk. This dramatically improves retrieval precision compared to
full-article indexing.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    """A single semantic chunk from a KB article."""

    chunk_id: str
    article_id: str
    title: str
    url: str
    category: str
    folder: str
    section_header: str
    content: str
    content_hash: str
    chunk_position: int
    chunk_total: int


def _find_section_headers(text: str) -> list[tuple[int, str]]:
    """Return (word_offset, header_text) pairs for detected section headings.

    Detects:
    - Markdown headings: ``# Heading``
    - Bold-only lines: ``**Heading**``
    """
    headers: list[tuple[int, str]] = []
    word_offset = 0
    for line in text.splitlines():
        line_stripped = line.strip()
        line_words = line_stripped.split()

        # Markdown heading
        if line_stripped.startswith("#"):
            header_text = re.sub(r"^#+\s*", "", line_stripped).strip()
            if header_text:
                headers.append((word_offset, header_text))

        # Short bold-only line (up to 10 words)
        elif (
            line_stripped.startswith("**")
            and line_stripped.endswith("**")
            and len(line_words) <= 10
        ):
            header_text = line_stripped.strip("*").strip()
            if header_text:
                headers.append((word_offset, header_text))

        word_offset += len(line_stripped.split())

    return headers


def _nearest_header_before(word_pos: int, headers: list[tuple[int, str]]) -> str:
    """Return the header text whose word offset is closest before word_pos."""
    result = ""
    for offset, text in headers:
        if offset <= word_pos:
            result = text
        else:
            break
    return result


def chunk_article(
    article: dict,
    chunk_words: int = 400,
    overlap_words: int = 80,
) -> list[TextChunk]:
    """Split a single article dict into overlapping TextChunks.

    Args:
        article: Article dict with keys: id, title, url, category, folder,
                 meta_description, content.
        chunk_words: Target words per chunk (≈520 tokens for English).
        overlap_words: Words of overlap between consecutive chunks.

    Returns:
        List of TextChunk objects; empty if no content.
    """
    article_id = str(article.get("id", ""))
    title = article.get("title", "")
    url = article.get("url", "")
    category = article.get("category", "")
    folder = article.get("folder", "")

    # Combine description + body
    content_parts: list[str] = []
    if article.get("meta_description"):
        content_parts.append(str(article["meta_description"]))
    if article.get("content"):
        content_parts.append(str(article["content"]))
    full_text = "\n\n".join(content_parts).strip()

    if not full_text:
        return []

    headers = _find_section_headers(full_text)
    words = full_text.split()
    total = len(words)

    if total == 0:
        return []

    # Build chunk boundaries (start, end) in word index space
    boundaries: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(start + chunk_words, total)
        boundaries.append((start, end))
        if end >= total:
            break
        start = end - overlap_words

    n = len(boundaries)
    chunks: list[TextChunk] = []

    for position, (s, e) in enumerate(boundaries):
        chunk_text = " ".join(words[s:e])
        section_header = _nearest_header_before(s, headers)
        content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()[:16]
        chunk_id = f"{article_id}__chunk_{position}"

        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                article_id=article_id,
                title=title,
                url=url,
                category=category,
                folder=folder,
                section_header=section_header,
                content=chunk_text,
                content_hash=content_hash,
                chunk_position=position,
                chunk_total=n,
            )
        )

    return chunks


def chunk_articles(
    articles: list[dict],
    chunk_words: int = 400,
    overlap_words: int = 80,
) -> list[TextChunk]:
    """Chunk a list of article dicts into TextChunks.

    Args:
        articles: List of article dicts from articles.json.
        chunk_words: Target words per chunk.
        overlap_words: Overlap between consecutive chunks.

    Returns:
        Flattened list of all chunks across all articles.
    """
    all_chunks: list[TextChunk] = []
    for article in articles:
        all_chunks.extend(chunk_article(article, chunk_words, overlap_words))
    return all_chunks
