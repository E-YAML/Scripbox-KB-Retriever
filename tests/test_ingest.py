"""Unit tests for ingest.py staged ingestion and rollback CLI."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import ingest


class TestIngest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.releases_path = Path(self.test_dir) / "releases"
        self.releases_path.mkdir(parents=True, exist_ok=True)

        self.orig_releases_root = ingest.RELEASES_ROOT
        self.orig_manifest_file = ingest.MANIFEST_FILE
        self.orig_current_file = ingest.CURRENT_FILE

        ingest.RELEASES_ROOT = self.releases_path
        ingest.MANIFEST_FILE = self.releases_path / "manifest.json"
        ingest.CURRENT_FILE = self.releases_path / "current.json"

        # Create dummy articles file
        self.sample_articles = [
            {"title": "Article 1", "url": "https://help.scripbox.com/1", "content": "Content of article one."},
            {"title": "Article 2", "url": "https://help.scripbox.com/2", "content": "Content of article two."},
        ]
        self.dummy_articles_file = Path(self.test_dir) / "dummy_articles.json"
        with open(self.dummy_articles_file, "w", encoding="utf-8") as f:
            json.dump(self.sample_articles, f)

    def tearDown(self):
        ingest.RELEASES_ROOT = self.orig_releases_root
        ingest.MANIFEST_FILE = self.orig_manifest_file
        ingest.CURRENT_FILE = self.orig_current_file
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_crawl_from_source(self):
        rel_id = ingest.cmd_crawl(source_file=str(self.dummy_articles_file))
        self.assertTrue((self.releases_path / rel_id / "articles.json").exists())

        manifest = ingest.load_manifest()
        self.assertIn(rel_id, manifest["releases"])
        self.assertEqual(manifest["releases"][rel_id]["status"], "staged")
        self.assertEqual(manifest["releases"][rel_id]["article_count"], 2)

    def test_diff_new_and_updated_articles(self):
        # 1. Stage release 1
        rel_1 = ingest.cmd_crawl(source_file=str(self.dummy_articles_file))

        # Mock build & promote release 1 to active
        b_info_1 = {
            "release_id": rel_1,
            "backend": "chroma",
            "collection_name": "coll_1",
            "chroma_dir": "./chroma_1",
            "article_count": 2,
            "chunk_count": 4,
            "embed_model": "test",
            "built_at": "2026-01-01",
        }
        ingest.atomic_write_json(self.releases_path / rel_1 / "build_info.json", b_info_1)
        ingest.cmd_promote(release_id=rel_1, smoke_test=False)

        # 2. Stage release 2 with changes: Article 1 updated, Article 2 kept, Article 3 added
        modified_articles = [
            {"title": "Article 1", "url": "https://help.scripbox.com/1", "content": "Updated content of article one."},
            {"title": "Article 2", "url": "https://help.scripbox.com/2", "content": "Content of article two."},
            {"title": "Article 3", "url": "https://help.scripbox.com/3", "content": "Content of article three."},
        ]
        articles_v2_file = Path(self.test_dir) / "articles_v2.json"
        with open(articles_v2_file, "w", encoding="utf-8") as f:
            json.dump(modified_articles, f)

        rel_2 = ingest.cmd_crawl(source_file=str(articles_v2_file))

        # Diff rel_2 against current active (rel_1)
        diff = ingest.cmd_diff(release_id=rel_2)
        counts = diff["counts"]

        self.assertEqual(counts["new"], 1)
        self.assertEqual(counts["updated"], 1)
        self.assertEqual(counts["deleted"], 0)
        self.assertEqual(counts["unchanged"], 1)

    def test_promote_and_rollback(self):
        # Stage & fake build release 1
        rel_1 = ingest.cmd_crawl(source_file=str(self.dummy_articles_file))
        b_info_1 = {
            "release_id": rel_1,
            "backend": "chroma",
            "collection_name": "coll_1",
            "chroma_dir": "./chroma_1",
            "article_count": 2,
            "chunk_count": 4,
            "embed_model": "test",
            "built_at": "2026-01-01",
        }
        ingest.atomic_write_json(self.releases_path / rel_1 / "build_info.json", b_info_1)
        ingest.cmd_promote(release_id=rel_1, smoke_test=False)

        current = ingest.load_current_release()
        self.assertEqual(current["release_id"], rel_1)
        self.assertIsNone(current["previous_release"])

        # Stage & fake build release 2
        rel_2 = ingest.cmd_crawl(source_file=str(self.dummy_articles_file))
        b_info_2 = {
            "release_id": rel_2,
            "backend": "chroma",
            "collection_name": "coll_2",
            "chroma_dir": "./chroma_2",
            "article_count": 2,
            "chunk_count": 4,
            "embed_model": "test",
            "built_at": "2026-01-02",
        }
        ingest.atomic_write_json(self.releases_path / rel_2 / "build_info.json", b_info_2)
        ingest.cmd_promote(release_id=rel_2, smoke_test=False)

        current_after_p2 = ingest.load_current_release()
        self.assertEqual(current_after_p2["release_id"], rel_2)
        self.assertEqual(current_after_p2["previous_release"], rel_1)

        # Rollback back to release 1
        rb = ingest.cmd_rollback()
        self.assertEqual(rb["release_id"], rel_1)

        current_after_rb = ingest.load_current_release()
        self.assertEqual(current_after_rb["release_id"], rel_1)

    def test_status_output(self):
        status = ingest.cmd_status()
        self.assertIn("current", status)
        self.assertIn("releases", status)


if __name__ == "__main__":
    unittest.main()
