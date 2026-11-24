"""Privacy-safe structured event logging for the Scripbox KB Retriever."""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

_logger = logging.getLogger("scripbox_kb")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)


def new_request_id() -> str:
    """Generate a new unique request ID."""
    import uuid
    return str(uuid.uuid4())


def _emit(event: dict[str, Any]) -> None:
    """Write a structured JSON event to stderr (no PII)."""
    _logger.info(json.dumps(event, default=str))


def log_app_startup(embed_model: str, providers_configured: list[str], vector_db: str) -> None:
    """Log application startup event."""
    _emit({
        "event": "app_startup",
        "embed_model": embed_model,
        "providers_configured": providers_configured,
        "vector_db": vector_db,
        "ts": time.time(),
    })


def log_retrieval(
    *,
    request_id: str,
    duration_ms: float,
    candidate_count: int,
    relevant_count: int,
    top_score: float | None,
    confidence_decision: str,
    rewrite_count: int = 0,
) -> None:
    """Log a retrieval event (no query text logged)."""
    _emit({
        "event": "retrieval",
        "request_id": request_id,
        "duration_ms": round(duration_ms, 1),
        "candidate_count": candidate_count,
        "relevant_count": relevant_count,
        "top_score": round(top_score, 4) if top_score is not None else None,
        "confidence_decision": confidence_decision,
        "rewrite_count": rewrite_count,
        "ts": time.time(),
    })


def log_rewrite(*, request_id: str, attempt: int) -> None:
    """Log a query rewrite event."""
    _emit({
        "event": "query_rewrite",
        "request_id": request_id,
        "attempt": attempt,
        "ts": time.time(),
    })


def log_generation(
    *,
    request_id: str,
    provider: str,
    fallback_reason: str | None,
    ttft_ms: float | None,
    total_ms: float,
    error_class: str | None = None,
) -> None:
    """Log a generation event (no prompt/completion text logged)."""
    _emit({
        "event": "generation",
        "request_id": request_id,
        "provider": provider,
        "fallback_reason": fallback_reason,
        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
        "total_ms": round(total_ms, 1),
        "error_class": error_class,
        "ts": time.time(),
    })


def log_response(
    *,
    request_id: str,
    status: str,
    rewrite_count: int,
    total_ms: float,
) -> None:
    """Log the final response status event."""
    _emit({
        "event": "response",
        "request_id": request_id,
        "status": status,
        "rewrite_count": rewrite_count,
        "total_ms": round(total_ms, 1),
        "ts": time.time(),
    })
