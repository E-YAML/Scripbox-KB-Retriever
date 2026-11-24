"""build_index.py — Scripbox KB Vector Index Builder (v2).

Reads articles.json, chunks articles into overlapping segments,
generates embeddings, and stores them in ChromaDB or Qdrant Cloud.

Usage:
    # ChromaDB (default, local dev):
    python build_index.py

    # Qdrant Cloud (production):
    VECTOR_DB_TYPE=qdrant QDRANT_URL=https://... QDRANT_API_KEY=... python build_index.py

Run after scraper.py whenever the knowledge base is updated.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer

from scripbox_kb.chunker import TextChunk, chunk_articles
from scripbox_kb.config import resolve_key

ARTICLES_FILE = "articles.json"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "scripbox_kb"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
CHUNK_WORDS = 400
OVERLAP_WORDS = 80

# Qdrant settings (resolved from secrets.toml, .env, or env vars)
VECTOR_DB_TYPE = resolve_key("VECTOR_DB_TYPE") or os.getenv("VECTOR_DB_TYPE", "chroma")
QDRANT_URL = resolve_key("QDRANT_URL") or os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = resolve_key("QDRANT_API_KEY") or os.getenv("QDRANT_API_KEY", "")
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 output dimension


def load_articles(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"'{path}' not found. Run scraper.py first.")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} articles from {path}")
    return data


def embed_chunks(chunks: list[TextChunk], model: SentenceTransformer) -> list[list[float]]:
    """Embed all chunks in batches. Returns list of embedding vectors."""
    texts = [c.content for c in chunks]
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(texts))
        print(f"  Embedding chunks {start + 1}–{end} of {len(texts)}...")
        batch = texts[start:end]
        embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())

    return all_embeddings


def build_chroma(chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
    """Write all chunks into a local ChromaDB collection."""
    import chromadb

    chroma_path = Path(CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(chroma_path))

    # Remove old collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Cleared existing ChromaDB collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 200
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch_chunks = chunks[start:end]
        batch_embs = embeddings[start:end]

        collection.add(
            ids=[c.chunk_id for c in batch_chunks],
            documents=[c.content for c in batch_chunks],
            embeddings=batch_embs,
            metadatas=[
                {
                    "title": c.title,
                    "url": c.url,
                    "category": c.category,
                    "folder": c.folder,
                    "section_header": c.section_header,
                    "article_id": c.article_id,
                    "chunk_position": c.chunk_position,
                    "chunk_total": c.chunk_total,
                    "content_hash": c.content_hash,
                }
                for c in batch_chunks
            ],
        )
        print(f"  Upserted chunks {start + 1}–{end} into ChromaDB")

    print(f"\nChromaDB index written to: {os.path.abspath(CHROMA_DIR)}")
    print(f"Total chunks indexed: {collection.count()}")


def build_qdrant(chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
    """Write all chunks into a Qdrant Cloud collection."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    if not QDRANT_URL:
        print("ERROR: QDRANT_URL environment variable is required for Qdrant mode.")
        sys.exit(1)

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

    # Recreate collection
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing Qdrant collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"  Created Qdrant collection '{COLLECTION_NAME}'")

    batch_size = 100  # Qdrant recommends smaller batches for cloud
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch_chunks = chunks[start:end]
        batch_embs = embeddings[start:end]

        points = [
            PointStruct(
                id=i + start,  # integer IDs for Qdrant
                vector=emb,
                payload={
                    "content": c.content,  # stored in payload for retrieval
                    "title": c.title,
                    "url": c.url,
                    "category": c.category,
                    "folder": c.folder,
                    "section_header": c.section_header,
                    "article_id": c.article_id,
                    "chunk_id": c.chunk_id,
                    "chunk_position": c.chunk_position,
                    "chunk_total": c.chunk_total,
                    "content_hash": c.content_hash,
                },
            )
            for i, (c, emb) in enumerate(zip(batch_chunks, batch_embs))
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  Upserted chunks {start + 1}–{end} into Qdrant")

    count = client.count(COLLECTION_NAME).count
    print(f"\nQdrant index ready. Total chunks: {count}")
    print(f"Collection: {COLLECTION_NAME} @ {QDRANT_URL}")


def main() -> None:
    t_start = time.perf_counter()

    articles = load_articles(ARTICLES_FILE)

    print(f"\nChunking {len(articles)} articles (chunk_words={CHUNK_WORDS}, overlap={OVERLAP_WORDS})...")
    chunks = chunk_articles(articles, chunk_words=CHUNK_WORDS, overlap_words=OVERLAP_WORDS)
    print(f"Generated {len(chunks)} chunks")

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    print("(First run downloads ~90 MB — please wait...)")
    model = SentenceTransformer(EMBED_MODEL)
    print("Model loaded")

    print(f"\nGenerating embeddings for {len(chunks)} chunks...")
    all_embeddings = embed_chunks(chunks, model)
    print(f"Embeddings done ({len(all_embeddings)} vectors)")

    print(f"\nWriting to vector DB (type={VECTOR_DB_TYPE})...")
    if VECTOR_DB_TYPE == "qdrant":
        build_qdrant(chunks, all_embeddings)
    else:
        build_chroma(chunks, all_embeddings)

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed:.1f}s — run python app.py (or streamlit run app.py) to serve.")


if __name__ == "__main__":
    main()
