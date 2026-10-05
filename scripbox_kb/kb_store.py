from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripbox_kb.config import AppConfig

try:
    import chromadb
    from sentence_transformers import SentenceTransformer
except ImportError:
    # Handle testing environment where these might not be installed
    pass


class KBError(Exception):
    """Base class for KB store exceptions."""


class KBUnavailableError(KBError):
    """Raised when the KB directory is missing or not ready."""


class CollectionError(KBError):
    """Raised when the ChromaDB collection cannot be loaded."""


class ModelLoadError(KBError):
    """Raised when the embedding model cannot be loaded."""


@dataclass
class KBStats:
    """Statistics about the knowledge base."""

    total: int = 0
    categories: int = 0
    category_list: list[str] = field(default_factory=list)


@dataclass
class RetrievalHit:
    """A single hit from a KB retrieval query."""

    title: str
    url: str
    category: str
    folder: str
    document: str
    score: float


class KBStore:
    """Manages the knowledge base resources (ChromaDB collection and embedding model)."""

    def __init__(
        self,
        chroma_dir: str,
        collection_name: str,
        embed_model_name: str,
        articles_file: str,
    ):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.embed_model_name = embed_model_name
        self._articles_file = articles_file

        self._collection: Any = None
        self._embed_model: Any = None
        self._status: str = "not_loaded"

    def load(self) -> None:
        """
        Loads the ChromaDB client, collection, and the sentence transformer model.

        Raises:
            KBUnavailableError: If the chroma directory does not exist.
            CollectionError: If there's an error getting the collection.
            ModelLoadError: If there's an error loading the embedding model.
        """
        db_path = Path(self.chroma_dir)
        if not db_path.exists():
            self._status = "chroma_missing"
            raise KBUnavailableError(f"Chroma DB directory missing: {self.chroma_dir}")

        try:
            client = chromadb.PersistentClient(path=str(db_path))
            self._collection = client.get_collection(self.collection_name)
        except Exception as exc:
            self._status = f"collection_error:{exc}"
            raise CollectionError(
                f"Failed to load collection {self.collection_name}: {exc}"
            ) from exc

        try:
            self._embed_model = SentenceTransformer(self.embed_model_name)
        except Exception as exc:
            self._status = f"model_error:{exc}"
            raise ModelLoadError(
                f"Failed to load embedding model {self.embed_model_name}: {exc}"
            ) from exc

        self._status = "ok"

    @property
    def status(self) -> str:
        """Current status of the KB store."""
        return self._status

    @property
    def is_ready(self) -> bool:
        """True if the collection and embedding model are successfully loaded."""
        return (
            self._collection is not None and self._embed_model is not None and self._status == "ok"
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """
        Retrieves top-k hits for the given query.

        Raises:
            KBUnavailableError: If the store is not loaded.
        """
        if not self.is_ready:
            raise KBUnavailableError("KB Store is not loaded. Call load() first.")

        qv = self._embed_model.encode(query).tolist()
        res = self._collection.query(
            query_embeddings=[qv],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if res.get("ids") and res["ids"][0]:
            for i in range(len(res["ids"][0])):
                score = float(1.0 - res["distances"][0][i])
                meta = res["metadatas"][0][i] or {}
                hits.append(
                    RetrievalHit(
                        title=meta.get("title", "Untitled"),
                        url=meta.get("url", "#"),
                        category=meta.get("category", ""),
                        folder=meta.get("folder", ""),
                        document=res["documents"][0][i],
                        score=score,
                    )
                )
        return hits

    def get_stats(self, articles_file: str | None = None) -> KBStats:
        """Reads the articles JSON file and returns stats."""
        filepath = articles_file if articles_file is not None else self._articles_file
        stats = KBStats()
        try:
            with open(filepath, encoding="utf-8") as f:
                articles = json.load(f)
            stats.total = len(articles)
            cats = sorted({a.get("category", "").strip() for a in articles if a.get("category")})
            stats.categories = len(cats)
            stats.category_list = cats
        except Exception:
            pass
        return stats

    def health_check(self) -> dict[str, bool]:
        """Returns the health status of the various KB components."""
        return {
            "db_exists": Path(self.chroma_dir).exists(),
            "collection_loaded": self._collection is not None,
            "model_loaded": self._embed_model is not None,
            "articles_file_exists": Path(self._articles_file).exists(),
        }


def create_kb_store(config: AppConfig) -> KBStore:
    """Creates a KBStore configured from an AppConfig."""
    return KBStore(
        chroma_dir=config.chroma_dir,
        collection_name=config.collection_name,
        embed_model_name=config.embed_model,
        articles_file=config.articles_file,
    )
