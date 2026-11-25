"""Tests for the LangGraph cyclic RAG graph."""

from unittest.mock import MagicMock, patch

from scripbox_kb.config import AppConfig
from scripbox_kb.kb_store import KBStore, RetrievalHit
from scripbox_kb.service import QueryService


def test_graph_cycles_on_low_scores_and_abstains():
    config = AppConfig(
        groq_api_key="test-key",
        gemini_api_key="test-key",
        retrieval_threshold=0.8,
        max_rewrites=2,
    )
    mock_kb = MagicMock(spec=KBStore)

    # Return low scoring hits below threshold
    low_hit = RetrievalHit(
        title="Unrelated",
        url="http://example.com/unrelated",
        category="Cat",
        folder="Folder",
        document="Irrelevant text",
        score=0.2,  # < 0.8 threshold
    )
    mock_kb.retrieve.return_value = [low_hit]

    with patch("scripbox_kb.providers.call_once", return_value="rewritten query"):
        service = QueryService(config, mock_kb)
        resp = service.handle_query("irrelevant question")

    # Should cycle max_rewrites (2) times then abstain
    assert resp.status == "abstained"
    assert resp.rewrite_count == 2
    assert mock_kb.retrieve.call_count == 3  # initial + 2 rewrites


def test_graph_succeeds_when_score_meets_threshold():
    config = AppConfig(
        groq_api_key="test-key",
        gemini_api_key="test-key",
        retrieval_threshold=0.5,
    )
    mock_kb = MagicMock(spec=KBStore)

    good_hit = RetrievalHit(
        title="KYC Guide",
        url="http://example.com/kyc",
        category="KYC",
        folder="Verification",
        document="KYC requires PAN and Aadhaar.",
        score=0.9,  # >= 0.5
    )
    mock_kb.retrieve.return_value = [good_hit]

    with patch("scripbox_kb.providers.stream_with_fallback") as mock_stream:
        mock_result = MagicMock()
        mock_result.provider = "groq"
        mock_result.stream = iter(["Answer text"])
        mock_result.fallback_reason = None
        mock_stream.return_value = mock_result

        service = QueryService(config, mock_kb)
        resp = service.handle_query("how to complete kyc")

    assert resp.status == "success"
    assert resp.rewrite_count == 0
    assert len(resp.sources) == 1
    assert resp.sources[0].title == "KYC Guide"
