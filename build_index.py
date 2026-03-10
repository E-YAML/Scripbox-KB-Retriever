"""
build_index.py — Scripbox KB Vector Index Builder
Reads articles.json, creates embeddings, and stores them in a
local ChromaDB vector database (persisted to ./chroma_db/).

Run ONCE after scraper.py:
    python build_index.py
"""
from __future__ import annotations  # Python 3.8+ compatible type hints

import json
import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ARTICLES_FILE   = "articles.json"
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "scripbox_kb"
EMBED_MODEL     = "all-MiniLM-L6-v2"
BATCH_SIZE      = 64


def load_articles(path: str) -> list:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"'{path}' not found. Please run scraper.py first."
        )
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} articles from {path}")
    return data


def build_document_text(article: dict) -> str:
    """Combine article fields into a single embeddable string."""
    parts = []
    if article.get("title"):
        # Repeat title for higher semantic weight
        parts.append(f"Title: {article['title']}")
        parts.append(f"Title: {article['title']}")
    if article.get("category"):
        parts.append(f"Category: {article['category']}")
    if article.get("folder"):
        parts.append(f"Section: {article['folder']}")
    if article.get("meta_description"):
        parts.append(article["meta_description"])
    if article.get("content"):
        parts.append(article["content"][:3000])
    return "\n".join(parts)


def main():
    articles = load_articles(ARTICLES_FILE)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    print("(First run will download ~90 MB — please wait...)")
    model = SentenceTransformer(EMBED_MODEL)
    print("Model loaded")

    docs      = []
    ids       = []
    metadatas = []

    for art in articles:
        docs.append(build_document_text(art))
        ids.append(art["id"])
        metadatas.append({
            "title":    art.get("title", "")[:500],
            "url":      art.get("url", ""),
            "category": art.get("category", ""),
            "folder":   art.get("folder", ""),
        })

    print(f"\nGenerating embeddings for {len(docs)} articles...")
    all_embeddings = []
    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start: start + BATCH_SIZE]
        end   = min(start + BATCH_SIZE, len(docs))
        print(f"  Embedding {start + 1}–{end}...")
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())
    print(f"Embeddings done ({len(all_embeddings)} vectors)")

    print(f"\nInitializing ChromaDB at: {CHROMA_DIR}")
    chroma_path = Path(CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(chroma_path))

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Cleared existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Upserting {len(docs)} documents...")
    chunk = 200
    for start in range(0, len(docs), chunk):
        end = min(start + chunk, len(docs))
        collection.add(
            documents=docs[start:end],
            embeddings=all_embeddings[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"\nIndexed {len(docs)} articles into '{COLLECTION_NAME}'")
    print(f"DB saved at: {os.path.abspath(CHROMA_DIR)}")
    print("\nNext: python retriever.py")


if __name__ == "__main__":
    main()
