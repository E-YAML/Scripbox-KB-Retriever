"""Tests for privacy-safe structured observability logging."""
from unittest.mock import patch

from scripbox_kb import observability


def test_new_request_id():
    req_id = observability.new_request_id()
    assert isinstance(req_id, str)
    assert len(req_id) > 10


@patch("scripbox_kb.observability._emit")
def test_log_retrieval(mock_emit):
    observability.log_retrieval(
        request_id="req-123",
        duration_ms=45.2,
        candidate_count=20,
        relevant_count=5,
        top_score=0.88,
        confidence_decision="sufficient",
        rewrite_count=1,
    )
    assert mock_emit.called
    event = mock_emit.call_args[0][0]
    assert event["event"] == "retrieval"
    assert event["request_id"] == "req-123"
    assert event["candidate_count"] == 20
    assert event["top_score"] == 0.88
    # Ensure no query text is emitted
    assert "query" not in event


@patch("scripbox_kb.observability._emit")
def test_log_response(mock_emit):
    observability.log_response(
        request_id="req-456",
        status="success",
        rewrite_count=0,
        total_ms=120.5,
    )
    assert mock_emit.called
    event = mock_emit.call_args[0][0]
    assert event["event"] == "response"
    assert event["status"] == "success"
