"""ingest.py — Staged KB Ingestion & Atomic Rollback CLI (Milestone 3).

Provides staged crawling, semantic diffing, isolated builds, atomic promotion,
and instant rollback for Scripbox Knowledge Base releases.

Directory layout:
    data/releases/
        current.json            # Atomically updated pointer to active release
        manifest.json           # Registry of all staged/active/archived releases
        <release_id>/
            articles.json       # Scraped knowledge base articles
            build_info.json     # Chunker & vector DB metadata
            chroma_db/          # Isolated ChromaDB directory (if chroma backend)

Usage:
    # 1. Crawl / stage articles into a new timestamped release
    python ingest.py crawl [--source articles.json]

    # 2. Diff staged release against the currently active release
    python ingest.py diff [--release <id>]

    # 3. Build vector index for the staged release
    python ingest.py build [--release <id>] [--backend chroma|qdrant]

    # 4. Validate and atomically promote staged release to active
    python ingest.py promote [--release <id>]

    # 5. Instant rollback to the previous active release
    python ingest.py rollback

    # 6. Show status of current release and release history
    python ingest.py status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripbox_kb.chunker import chunk_articles
from scripbox_kb.config import resolve_key

RELEASES_ROOT = Path("data/releases")
MANIFEST_FILE = RELEASES_ROOT / "manifest.json"
CURRENT_FILE = RELEASES_ROOT / "current.json"


# ── Helpers & State Management ────────────────────────────────────────────────


def ensure_releases_dir() -> Path:
    """Ensure data/releases directory exists and return it."""
    RELEASES_ROOT.mkdir(parents=True, exist_ok=True)
    return RELEASES_ROOT


def load_manifest() -> dict[str, Any]:
    """Load release registry manifest."""
    ensure_releases_dir()
    if not MANIFEST_FILE.exists():
        return {"releases": {}}
    try:
        with open(MANIFEST_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"releases": {}}


def save_manifest(manifest: dict[str, Any]) -> None:
    """Save release registry manifest atomically."""
    ensure_releases_dir()
    tmp_path = MANIFEST_FILE.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, MANIFEST_FILE)


def load_current_release() -> dict[str, Any] | None:
    """Load metadata for the currently active release, if any."""
    if not CURRENT_FILE.exists():
        return None
    try:
        with open(CURRENT_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def atomic_write_json(target_path: Path, data: dict[str, Any]) -> None:
    """Write dictionary to target path atomically using a temporary file."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, target_path)


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def get_latest_staged_release(manifest: dict[str, Any]) -> str | None:
    """Find the most recent release with status 'staged' or 'built'."""
    releases = manifest.get("releases", {})
    sorted_ids = sorted(releases.keys(), reverse=True)
    for rid in sorted_ids:
        if releases[rid].get("status") in ("staged", "built"):
            return rid
    return sorted_ids[0] if sorted_ids else None


# ── Subcommand 1: Crawl / Stage ───────────────────────────────────────────────


def cmd_crawl(source_file: str | None = None) -> str:
    """Stage articles into a new release directory.

    If source_file is provided, copies from that file. Otherwise runs scraper.
    """
    ensure_releases_dir()
    release_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    release_dir = RELEASES_ROOT / release_id
    release_dir.mkdir(parents=True, exist_ok=True)
    target_articles_file = release_dir / "articles.json"

    if source_file:
        src = Path(source_file)
        if not src.exists():
            raise FileNotFoundError(f"Source file '{source_file}' not found.")
        with open(src, encoding="utf-8") as f:
            articles = json.load(f)
        with open(target_articles_file, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        print(
            f"[CRAWL] Staged {len(articles)} articles from '{source_file}' -> {target_articles_file}"
        )
    else:
        print("[CRAWL] Scraping articles from https://help.scripbox.com...")
        import scraper

        folders = scraper.discover_folder_urls()
        discovered = scraper.discover_article_urls(folders)
        articles = []
        for i, (url, category, folder) in enumerate(discovered, 1):
            art = scraper.scrape_article(url, category=category, folder=folder)
            if art:
                articles.append(art)
        with open(target_articles_file, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        print(f"[CRAWL] Scraped {len(articles)} articles -> {target_articles_file}")

    manifest = load_manifest()
    manifest.setdefault("releases", {})[release_id] = {
        "release_id": release_id,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "staged",
        "article_count": len(articles),
        "articles_file": str(target_articles_file).replace("\\", "/"),
    }
    save_manifest(manifest)
    print(f"[CRAWL] Created release: {release_id} (Status: staged)")
    return release_id


# ── Subcommand 2: Diff ────────────────────────────────────────────────────────


def cmd_diff(release_id: str | None = None) -> dict[str, Any]:
    """Diff a staged release against the currently active release."""
    manifest = load_manifest()
    if not release_id:
        release_id = get_latest_staged_release(manifest)
        if not release_id:
            print("[DIFF] No staged releases found to diff.")
            return {"error": "no_staged_release"}

    staged_path = RELEASES_ROOT / release_id / "articles.json"
    if not staged_path.exists():
        raise FileNotFoundError(f"Release '{release_id}' articles not found at {staged_path}")

    with open(staged_path, encoding="utf-8") as f:
        staged_articles = json.load(f)

    current = load_current_release()
    current_articles = []
    if current and "articles_file" in current:
        cur_file = Path(current["articles_file"])
        if cur_file.exists():
            with open(cur_file, encoding="utf-8") as f:
                current_articles = json.load(f)

    # Key by URL if present, otherwise by title
    def art_key(a: dict) -> str:
        return a.get("url") or a.get("title", "")

    cur_map = {art_key(a): a for a in current_articles}
    staged_map = {art_key(a): a for a in staged_articles}

    new_articles = []
    deleted_articles = []
    updated_articles = []
    unchanged_articles = []

    for k, s_art in staged_map.items():
        if k not in cur_map:
            new_articles.append(s_art.get("title", k))
        else:
            c_art = cur_map[k]
            s_hash = compute_content_hash(s_art.get("content", ""))
            c_hash = compute_content_hash(c_art.get("content", ""))
            if s_hash != c_hash:
                updated_articles.append(s_art.get("title", k))
            else:
                unchanged_articles.append(s_art.get("title", k))

    for k, c_art in cur_map.items():
        if k not in staged_map:
            deleted_articles.append(c_art.get("title", k))

    diff_summary = {
        "staged_release": release_id,
        "active_release": current.get("release_id") if current else None,
        "counts": {
            "staged_total": len(staged_articles),
            "active_total": len(current_articles),
            "new": len(new_articles),
            "updated": len(updated_articles),
            "deleted": len(deleted_articles),
            "unchanged": len(unchanged_articles),
        },
        "new_titles": new_articles[:10],
        "updated_titles": updated_articles[:10],
        "deleted_titles": deleted_articles[:10],
    }

    print(f"\n{'=' * 60}")
    print(
        f"Release Diff: Staged [{release_id}] vs Active [{diff_summary['active_release'] or 'None'}]"
    )
    print(f"{'=' * 60}")
    print(f"  Total Articles in Staged:  {len(staged_articles)}")
    print(f"  Total Articles in Active:  {len(current_articles)}")
    print(f"  + New:                     {len(new_articles)}")
    print(f"  ~ Updated:                 {len(updated_articles)}")
    print(f"  - Deleted:                 {len(deleted_articles)}")
    print(f"  = Unchanged:               {len(unchanged_articles)}")

    if new_articles:
        print("\n  Sample New:")
        for t in new_articles[:5]:
            print(f"    + {t}")
    if updated_articles:
        print("\n  Sample Updated:")
        for t in updated_articles[:5]:
            print(f"    ~ {t}")
    if deleted_articles:
        print("\n  Sample Deleted:")
        for t in deleted_articles[:5]:
            print(f"    - {t}")
    print(f"{'=' * 60}\n")

    return diff_summary


# ── Subcommand 3: Build ───────────────────────────────────────────────────────


def cmd_build(
    release_id: str | None = None,
    backend: str | None = None,
    embed_model: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Chunk and embed articles for the specified release into an isolated collection/directory."""
    manifest = load_manifest()
    if not release_id:
        release_id = get_latest_staged_release(manifest)
        if not release_id:
            raise ValueError("No staged release available to build.")

    release_dir = RELEASES_ROOT / release_id
    articles_path = release_dir / "articles.json"
    if not articles_path.exists():
        raise FileNotFoundError(f"Articles not found for release '{release_id}' at {articles_path}")

    with open(articles_path, encoding="utf-8") as f:
        articles = json.load(f)

    backend = backend or resolve_key("VECTOR_DB_TYPE") or "chroma"
    print(f"[BUILD] Building release {release_id} using backend '{backend}'...")

    chunks = chunk_articles(articles, chunk_words=400, overlap_words=80)
    print(
        f"[BUILD] Generated {len(chunks)} overlapping semantic chunks from {len(articles)} articles."
    )

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(embed_model)
    texts = [c.content for c in chunks]
    embeddings = encoder.encode(texts, batch_size=64, show_progress_bar=False).tolist()

    collection_name = f"scripbox_kb_{release_id}"
    chroma_dir = ""

    if backend == "chroma":
        import chromadb

        chroma_dir_path = release_dir / "chroma_db"
        chroma_dir = str(chroma_dir_path).replace("\\", "/")
        client = chromadb.PersistentClient(path=str(chroma_dir_path))
        try:
            client.delete_collection("scripbox_kb")
        except Exception:
            pass
        collection = client.create_collection("scripbox_kb", metadata={"hnsw:space": "cosine"})

        # Upsert in batches of 64
        for start in range(0, len(chunks), 64):
            end = min(start + 64, len(chunks))
            b_chunks = chunks[start:end]
            collection.upsert(
                ids=[f"{c.article_id}_chunk_{c.chunk_position}" for c in b_chunks],
                embeddings=embeddings[start:end],
                documents=[c.content for c in b_chunks],
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
                    for c in b_chunks
                ],
            )
        print(f"[BUILD] ChromaDB collection populated at: {chroma_dir}")
    elif backend == "qdrant":
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        qdrant_url = resolve_key("QDRANT_URL")
        qdrant_key = resolve_key("QDRANT_API_KEY")
        if not qdrant_url:
            raise ValueError("QDRANT_URL required to build Qdrant collection.")

        client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=30)
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
        )
        for start in range(0, len(chunks), 100):
            end = min(start + 100, len(chunks))
            pts = [
                PointStruct(
                    id=i + start,
                    vector=embeddings[i + start],
                    payload={
                        "content": c.content,
                        "title": c.title,
                        "url": c.url,
                        "category": c.category,
                        "folder": c.folder,
                        "section_header": c.section_header,
                        "article_id": c.article_id,
                        "chunk_position": c.chunk_position,
                        "chunk_total": c.chunk_total,
                        "content_hash": c.content_hash,
                    },
                )
                for i, c in enumerate(chunks[start:end])
            ]
            client.upsert(collection_name=collection_name, points=pts)
        print(f"[BUILD] Qdrant collection '{collection_name}' populated.")
    else:
        raise ValueError(f"Unknown backend: {backend}")

    build_info = {
        "release_id": release_id,
        "backend": backend,
        "collection_name": collection_name if backend == "qdrant" else "scripbox_kb",
        "chroma_dir": chroma_dir,
        "article_count": len(articles),
        "chunk_count": len(chunks),
        "embed_model": embed_model,
        "built_at": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(release_dir / "build_info.json", build_info)

    manifest = load_manifest()
    rel_entry = manifest.setdefault("releases", {}).setdefault(release_id, {})
    rel_entry.update(
        {
            "status": "built",
            "backend": backend,
            "chunk_count": len(chunks),
            "built_at": build_info["built_at"],
            "collection_name": build_info["collection_name"],
            "chroma_dir": chroma_dir,
        }
    )
    save_manifest(manifest)
    print(f"[BUILD] Successfully built release '{release_id}'!")
    return build_info


# ── Subcommand 4: Promote ─────────────────────────────────────────────────────


def cmd_promote(release_id: str | None = None, smoke_test: bool = True) -> dict[str, Any]:
    """Validate and atomically promote a built release to active."""
    manifest = load_manifest()
    if not release_id:
        release_id = get_latest_staged_release(manifest)
        if not release_id:
            raise ValueError("No staged/built release to promote.")

    release_dir = RELEASES_ROOT / release_id
    build_info_path = release_dir / "build_info.json"
    if not build_info_path.exists():
        raise FileNotFoundError(
            f"Release '{release_id}' has not been built yet (missing build_info.json). Run 'python ingest.py build' first."
        )

    with open(build_info_path, encoding="utf-8") as f:
        build_info = json.load(f)

    # Smoke test verification
    if smoke_test:
        print(f"[PROMOTE] Running smoke test on release '{release_id}'...")
        from scripbox_kb.config import AppConfig
        from scripbox_kb.kb_store import KBStore

        test_config = AppConfig(
            vector_db_type=build_info["backend"],
            collection_name=build_info["collection_name"],
            chroma_dir=build_info.get("chroma_dir") or "./chroma_db",
            qdrant_url=resolve_key("QDRANT_URL"),
            qdrant_api_key=resolve_key("QDRANT_API_KEY"),
            top_k=2,
        )
        store = KBStore(test_config)
        hits = store.retrieve("KYC investment account", top_k=2)
        if not hits:
            raise RuntimeError(
                f"Smoke test failed: retrieval returned 0 hits for release '{release_id}'. Aborting promotion."
            )
        print(f"[PROMOTE] Smoke test passed! Retrieved {len(hits)} hits (top: '{hits[0].title}')")

    current = load_current_release()
    previous_id = current.get("release_id") if current else None

    new_current = {
        "release_id": release_id,
        "promoted_at": datetime.now(UTC).isoformat(),
        "previous_release": previous_id,
        "backend": build_info["backend"],
        "collection_name": build_info["collection_name"],
        "chroma_dir": build_info["chroma_dir"],
        "articles_file": str(release_dir / "articles.json").replace("\\", "/"),
        "article_count": build_info["article_count"],
        "chunk_count": build_info["chunk_count"],
    }

    # Atomic pointer swap
    atomic_write_json(CURRENT_FILE, new_current)

    # Update manifest statuses
    for rid, rdata in manifest.get("releases", {}).items():
        if rid == release_id:
            rdata["status"] = "active"
            rdata["promoted_at"] = new_current["promoted_at"]
        elif rdata.get("status") == "active":
            rdata["status"] = "archived"

    save_manifest(manifest)
    print(f"\n[PROMOTE] ✅ Release '{release_id}' is now ACTIVE!")
    print(f"          Previous active release: {previous_id or 'None'}")
    return new_current


# ── Subcommand 5: Rollback ────────────────────────────────────────────────────


def cmd_rollback() -> dict[str, Any]:
    """Roll back active release to previous release instantaneously."""
    current = load_current_release()
    if not current:
        raise RuntimeError("No currently active release found to roll back from.")

    prev_id = current.get("previous_release")
    if not prev_id:
        raise RuntimeError("Cannot roll back: current release has no 'previous_release' recorded.")

    prev_dir = RELEASES_ROOT / prev_id
    prev_build_info = prev_dir / "build_info.json"
    if not prev_build_info.exists():
        raise FileNotFoundError(
            f"Previous release directory or build_info not found at {prev_build_info}"
        )

    with open(prev_build_info, encoding="utf-8") as f:
        build_info = json.load(f)

    active_id = current["release_id"]
    new_current = {
        "release_id": prev_id,
        "promoted_at": datetime.now(UTC).isoformat(),
        "previous_release": active_id,  # allows rolling forward if needed
        "backend": build_info["backend"],
        "collection_name": build_info["collection_name"],
        "chroma_dir": build_info["chroma_dir"],
        "articles_file": str(prev_dir / "articles.json").replace("\\", "/"),
        "article_count": build_info["article_count"],
        "chunk_count": build_info["chunk_count"],
    }

    atomic_write_json(CURRENT_FILE, new_current)

    manifest = load_manifest()
    for rid, rdata in manifest.get("releases", {}).items():
        if rid == prev_id:
            rdata["status"] = "active"
        elif rid == active_id:
            rdata["status"] = "archived"

    save_manifest(manifest)
    print(f"\n[ROLLBACK] ⏪ Rolled back active release from '{active_id}' -> '{prev_id}'.")
    return new_current


# ── Subcommand 6: Status ──────────────────────────────────────────────────────


def cmd_status() -> dict[str, Any]:
    """Display status of current active release and release history."""
    current = load_current_release()
    manifest = load_manifest()
    releases = manifest.get("releases", {})

    print(f"\n{'=' * 65}")
    print("Scripbox KB Ingestion Status")
    print(f"{'=' * 65}")
    if current:
        print(f"ACTIVE RELEASE:     {current.get('release_id')}")
        print(f"  Backend:          {current.get('backend')}")
        print(f"  Collection/Dir:   {current.get('collection_name') or current.get('chroma_dir')}")
        print(f"  Articles:         {current.get('article_count')}")
        print(f"  Chunks:           {current.get('chunk_count')}")
        print(f"  Promoted At:      {current.get('promoted_at')}")
        print(f"  Previous Release: {current.get('previous_release') or 'None'}")
    else:
        print("ACTIVE RELEASE:     None (using base legacy index)")

    print(f"\nRelease Registry ({len(releases)} registered):")
    print(f"{'Release ID':<22} {'Status':<10} {'Articles':<10} {'Chunks':<10} {'Backend':<10}")
    print("-" * 65)
    for rid in sorted(releases.keys(), reverse=True):
        r = releases[rid]
        status = r.get("status", "unknown")
        art_cnt = str(r.get("article_count", "-"))
        chk_cnt = str(r.get("chunk_count", "-"))
        bkend = str(r.get("backend", "-"))
        mark = "▶" if current and current.get("release_id") == rid else " "
        print(f"{mark} {rid:<20} {status:<10} {art_cnt:<10} {chk_cnt:<10} {bkend:<10}")
    print(f"{'=' * 65}\n")

    return {"current": current, "releases": releases}


# ── CLI Entrypoint ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Scripbox KB Staged Ingestion & Rollback CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # crawl
    p_crawl = subparsers.add_parser("crawl", help="Stage new release from scraper or file")
    p_crawl.add_argument("--source", help="Path to existing articles.json (skips network crawl)")

    # diff
    p_diff = subparsers.add_parser("diff", help="Diff staged release against active release")
    p_diff.add_argument("--release", help="Release ID to diff (defaults to latest staged)")

    # build
    p_build = subparsers.add_parser("build", help="Chunk and index staged release")
    p_build.add_argument("--release", help="Release ID to build (defaults to latest staged)")
    p_build.add_argument("--backend", choices=["chroma", "qdrant"], help="Vector DB backend")
    p_build.add_argument(
        "--embed-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model"
    )

    # promote
    p_promote = subparsers.add_parser("promote", help="Validate and promote release to active")
    p_promote.add_argument("--release", help="Release ID to promote (defaults to latest built)")
    p_promote.add_argument(
        "--skip-smoke-test", action="store_true", help="Skip smoke test before promoting"
    )

    # rollback
    subparsers.add_parser("rollback", help="Instant rollback to previous release")

    # status
    subparsers.add_parser("status", help="Show active release and registry status")

    args = parser.parse_args()

    if args.command == "crawl":
        cmd_crawl(source_file=args.source)
    elif args.command == "diff":
        cmd_diff(release_id=args.release)
    elif args.command == "build":
        cmd_build(release_id=args.release, backend=args.backend, embed_model=args.embed_model)
    elif args.command == "promote":
        cmd_promote(release_id=args.release, smoke_test=not args.skip_smoke_test)
    elif args.command == "rollback":
        cmd_rollback()
    elif args.command == "status":
        cmd_status()


if __name__ == "__main__":
    main()
