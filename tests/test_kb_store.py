import json

import pytest

from scripbox_kb.kb_store import (
    KBStats,
    KBStore,
    KBUnavailableError,
    RetrievalHit,
)


def test_retrieval_hit_dataclass():
    hit = RetrievalHit(
        title="Test",
        url="http://test",
        category="Cat",
        folder="Folder",
        document="Doc",
        score=0.9,
    )
    assert hit.title == "Test"
    assert hit.score == 0.9


def test_kb_stats_dataclass():
    stats = KBStats(total=10, categories=2, category_list=["A", "B"])
    assert stats.total == 10
    assert stats.categories == 2
    assert stats.category_list == ["A", "B"]


def test_kb_store_not_loaded():
    store = KBStore("dir", "col", "mod", "art.json")
    assert not store.is_ready
    assert store.status == "not_loaded"


def test_kb_store_load_missing_dir():
    store = KBStore("non_existent_dir_123", "col", "mod", "art.json")
    with pytest.raises(KBUnavailableError):
        store.load()


def test_retrieve_not_loaded():
    store = KBStore("dir", "col", "mod", "art.json")
    with pytest.raises(KBUnavailableError):
        store.retrieve("query")


def test_get_stats_from_file(tmp_path):
    articles_file = tmp_path / "articles.json"
    articles_data = [
        {"title": "1", "category": "A"},
        {"title": "2", "category": "B"},
        {"title": "3", "category": "A"},
    ]
    articles_file.write_text(json.dumps(articles_data))

    store = KBStore("dir", "col", "mod", str(articles_file))
    stats = store.get_stats()
    assert stats.total == 3
    assert stats.categories == 2
    assert stats.category_list == ["A", "B"]


def test_get_stats_missing_file():
    store = KBStore("dir", "col", "mod", "missing_articles.json")
    stats = store.get_stats()
    assert stats.total == 0
    assert stats.categories == 0
    assert stats.category_list == []


def test_health_check_not_loaded(tmp_path):
    store = KBStore(str(tmp_path), "col", "mod", "art.json")
    health = store.health_check()
    assert health["db_exists"] is True
    assert health["collection_loaded"] is False
    assert health["model_loaded"] is False
    assert health["articles_file_exists"] is False
