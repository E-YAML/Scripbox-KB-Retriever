"""Knowledge base store: ChromaDB and Qdrant backends + embedding model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripbox_kb.config import AppConfig

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass  # handled at load time


# ─── Exceptions ───────────────────────────────────────────────────────────────


class KBError(Exception):
    """Base class for KB store exceptions."""


class KBUnavailableError(KBError):
    """Raised when the KB directory is missing or not ready."""


class CollectionError(KBError):
    """Raised when the vector collection cannot be loaded."""


class ModelLoadError(KBError):
    """Raised when the embedding model cannot be loaded."""


# ─── Data models ──────────────────────────────────────────────────────────────


@dataclass
class KBStats:
    """Statistics about the knowledge base."""

    total: int = 0
    categories: int = 0
    category_list: list[str] = field(default_factory=list)


@dataclass
class RetrievalHit:
    """A single hit returned from a KB retrieval query."""

    title: str
    url: str
    category: str
    folder: str
    document: str
    score: float


# ─── Backend abstractions ─────────────────────────────────────────────────────


class _ChromaBackend:
    """ChromaDB PersistentClient retrieval backend."""

    def __init__(self, chroma_dir: str, collection_name: str) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=chroma_dir)
        self._collection = client.get_collection(collection_name)

    def search(
        self, query_vector: list[float], top_k: int
    ) -> list[tuple[str, dict[str, Any], float]]:
        """Return (document, metadata, distance) tuples."""
        res = self._collection.query(
            query_embeddings=[query_vector],  # type: ignore[arg-type]
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        results: list[tuple[str, dict[str, Any], float]] = []
        ids_list = res.get("ids")
        docs_list = res.get("documents")
        metas_list = res.get("metadatas")
        dists_list = res.get("distances")
        if ids_list and ids_list[0] and docs_list and metas_list and dists_list:
            for i in range(len(ids_list[0])):
                doc = str(docs_list[0][i])
                meta_item = metas_list[0][i]
                meta = dict(meta_item) if isinstance(meta_item, dict) else {}
                dist = float(dists_list[0][i])
                results.append((doc, meta, dist))
        return results

    def count(self) -> int:
        return self._collection.count()


class _QdrantBackend:
    """Qdrant Cloud retrieval backend."""

    def __init__(self, url: str, api_key: str, collection_name: str) -> None:
        from qdrant_client import QdrantClient

        self._client = QdrantClient(url=url, api_key=api_key, timeout=10)
        self._collection_name = collection_name

    def search(
        self, query_vector: list[float], top_k: int
    ) -> list[tuple[str, dict[str, Any], float]]:
        """Return (document, metadata, distance) tuples.

        Qdrant returns cosine *similarity* scores (1 = identical).
        We convert to distance = 1 - similarity for uniform handling.
        """
        # Modern Qdrant client uses query_points, older uses search
        if hasattr(self._client, "query_points"):
            res = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
            points = res.points
        else:
            search_fn = getattr(self._client, "search")  # noqa: B009
            points = search_fn(
                collection_name=self._collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )

        output: list[tuple[str, dict[str, Any], float]] = []
        for r in points:
            payload = dict(r.payload or {})
            doc = payload.get("content", payload.get("document", ""))
            dist = float(1.0 - r.score)  # similarity → distance
            output.append((doc, payload, dist))
        return output

    def count(self) -> int:
        return self._client.count(self._collection_name).count


# ─── Main store ───────────────────────────────────────────────────────────────


class KBStore:
    """Manages vector DB backend + embedding model for the knowledge base."""

    def __init__(
        self,
        chroma_dir: str,
        collection_name: str,
        embed_model_name: str,
        articles_file: str,
        vector_db_type: str = "chroma",
        qdrant_url: str = "",
        qdrant_api_key: str = "",
    ) -> None:
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.embed_model_name = embed_model_name
        self._articles_file = articles_file
        self._vector_db_type = vector_db_type
        self._qdrant_url = qdrant_url
        self._qdrant_api_key = qdrant_api_key

        self._backend: _ChromaBackend | _QdrantBackend | None = None
        self._embed_model: Any = None
        self._status: str = "not_loaded"

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the vector DB backend and embedding model.

        Raises:
            KBUnavailableError: ChromaDB directory missing (chroma mode).
            CollectionError: Cannot connect or fetch the collection.
            ModelLoadError: Cannot load the embedding model.
        """
        # 1. Vector DB backend
        if self._vector_db_type == "qdrant":
            if not self._qdrant_url:
                self._status = "qdrant_missing_url"
                raise KBUnavailableError("QDRANT_URL is not configured.")
            try:
                self._backend = _QdrantBackend(
                    self._qdrant_url, self._qdrant_api_key, self.collection_name
                )
            except Exception as exc:
                self._status = f"qdrant_error:{exc}"
                raise CollectionError(f"Failed to connect to Qdrant: {exc}") from exc
        else:
            db_path = Path(self.chroma_dir)
            if not db_path.exists():
                self._status = "chroma_missing"
                raise KBUnavailableError(f"Chroma DB directory missing: {self.chroma_dir}")
            try:
                self._backend = _ChromaBackend(str(db_path), self.collection_name)
            except Exception as exc:
                self._status = f"collection_error:{exc}"
                raise CollectionError(
                    f"Failed to load collection {self.collection_name}: {exc}"
                ) from exc

        # 2. Embedding model
        try:
            self._embed_model = SentenceTransformer(self.embed_model_name)
        except Exception as exc:
            self._status = f"model_error:{exc}"
            raise ModelLoadError(
                f"Failed to load embedding model {self.embed_model_name}: {exc}"
            ) from exc

        self._status = "ok"

    # ── Status ─────────────────────────────────────────────────────────────────

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_ready(self) -> bool:
        return self._backend is not None and self._embed_model is not None and self._status == "ok"

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        """Embed query and retrieve top-k hits from the vector backend.

        Args:
            query: User query string.
            top_k: Number of results to return.

        Returns:
            List of RetrievalHit objects sorted by descending score.

        Raises:
            KBUnavailableError: Store not loaded.
        """
        if not self.is_ready:
            raise KBUnavailableError("KB Store is not loaded. Call load() first.")

        qv: list[float] = self._embed_model.encode(query).tolist()
        raw = self._backend.search(qv, top_k)  # type: ignore[union-attr]

        hits: list[RetrievalHit] = []
        for doc, meta, distance in raw:
            score = float(1.0 - distance)
            hits.append(
                RetrievalHit(
                    title=meta.get("title", "Untitled"),
                    url=meta.get("url", "#"),
                    category=meta.get("category", ""),
                    folder=meta.get("folder", ""),
                    document=doc,
                    score=score,
                )
            )
        return hits

    # ── Stats / health ─────────────────────────────────────────────────────────

    def get_stats(self, articles_file: str | None = None) -> KBStats:
        """Read articles.json and return aggregate stats."""
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
        """Return health status for all KB components."""
        db_exists = (
            Path(self.chroma_dir).exists()
            if self._vector_db_type != "qdrant"
            else bool(self._qdrant_url)
        )
        return {
            "db_exists": db_exists,
            "collection_loaded": self._backend is not None,
            "db_connected": self._backend is not None,
            "model_loaded": self._embed_model is not None,
            "articles_file_exists": Path(self._articles_file).exists(),
        }

    # ── Internal access (used by app.py for display) ───────────────────────────

    @property
    def _collection(self) -> Any:
        """Compatibility shim: return backend for count access."""
        return self._backend


# ─── Factory ─────────────────────────────────────────────────────────────────


def create_kb_store(config: AppConfig) -> KBStore:
    """Create a KBStore from an AppConfig."""
    return KBStore(
        chroma_dir=config.chroma_dir,
        collection_name=config.collection_name,
        embed_model_name=config.embed_model,
        articles_file=config.articles_file,
        vector_db_type=config.vector_db_type,
        qdrant_url=config.qdrant_url,
        qdrant_api_key=config.qdrant_api_key,
    )
