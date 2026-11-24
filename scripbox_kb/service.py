"""Service orchestration: wraps the LangGraph cyclic RAG pipeline.

``QueryService`` maintains backward-compatible ``ServiceResponse`` output so
``app.py`` requires minimal changes.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from uuid import uuid4

from scripbox_kb import observability
from scripbox_kb.config import AppConfig
from scripbox_kb.graph import build_rag_graph
from scripbox_kb.kb_store import KBStore, RetrievalHit


@dataclass
class ServiceResponse:
    """Standardised response from the query service (unchanged interface)."""

    status: str  # success | abstained | validation_error | retrieval_failure | provider_failure
    answer_stream: Iterator[str] | None = None
    answer_text: str | None = None
    sources: list[RetrievalHit] = field(default_factory=list)
    provider_used: str | None = None
    fallback_reason: str | None = None
    error_message: str | None = None
    # Graph-specific metadata (new, optional)
    rewrite_count: int = 0
    rewritten_query: str | None = None


class QueryService:
    """Handles user queries through the cyclic LangGraph RAG pipeline."""

    def __init__(self, config: AppConfig, kb_store: KBStore) -> None:
        """Initialise and compile the RAG graph and concurrency guard.

        Args:
            config: Application configuration.
            kb_store: Loaded knowledge-base store.
        """
        self._config = config
        self._graph = build_rag_graph(config, kb_store)
        self._semaphore = threading.BoundedSemaphore(max(1, config.max_concurrency))

    def handle_query(self, query: str) -> ServiceResponse:
        """Process a query through the graph and return a ServiceResponse.

        The graph may cycle through retrieve → grade → rewrite → retrieve
        up to ``config.max_rewrites`` times before abstaining. Concurrency
        is bounded to prevent burning provider rate limits during traffic bursts.

        Args:
            query: Raw user query string.

        Returns:
            ServiceResponse with either a streaming answer or an error/abstain state.
        """
        # Guard concurrency: acquire a processing slot or return busy
        acquired = self._semaphore.acquire(timeout=self._config.concurrency_timeout)
        if not acquired:
            return ServiceResponse(
                status="provider_failure",
                error_message="The assistant is currently assisting other users. Please try again in a moment.",
            )

        try:
            request_id = str(uuid4())
            ts_start = time.perf_counter()

            initial_state: dict = {
                "request_id": request_id,
                "query": query,
                "ts_start": ts_start,
                "validation_error": None,
                "effective_query": query,
                "retrieved_hits": [],
                "graded_hits": [],
                "rewrite_count": 0,
                "rewritten_query": None,
                "answer_stream": None,
                "answer_text": None,
                "provider_used": None,
                "fallback_reason": None,
                "sources": [],
                "status": "unknown",
                "error_message": None,
                "retrieval_ms": None,
                "generation_ms": None,
            }

            final_state: dict = self._graph.invoke(initial_state)

            total_ms = (time.perf_counter() - ts_start) * 1000
            status = final_state.get("status", "unknown")
            rewrite_count = final_state.get("rewrite_count", 0)

            observability.log_response(
                request_id=request_id,
                status=status,
                rewrite_count=rewrite_count,
                total_ms=total_ms,
            )

            return ServiceResponse(
                status=status,
                answer_stream=final_state.get("answer_stream"),
                answer_text=final_state.get("answer_text"),
                sources=final_state.get("sources", []),
                provider_used=final_state.get("provider_used"),
                fallback_reason=final_state.get("fallback_reason"),
                error_message=final_state.get("error_message"),
                rewrite_count=rewrite_count,
                rewritten_query=final_state.get("rewritten_query"),
            )
        finally:
            self._semaphore.release()

    def _sanitize_error(self, exc: Exception) -> str:
        """Sanitize exception messages for user-facing output.

        Args:
            exc: The exception to sanitize.

        Returns:
            A user-safe error message string.
        """
        from scripbox_kb.providers import sanitize_error

        return sanitize_error(exc)
