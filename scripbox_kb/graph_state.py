"""LangGraph state definition for the cyclic RAG pipeline."""
from __future__ import annotations

import operator
from collections.abc import Iterator
from typing import Annotated

from typing_extensions import TypedDict

from scripbox_kb.kb_store import RetrievalHit


class RAGState(TypedDict):
    """Full shared state flowing through every node in the RAG graph.

    Fields annotated with ``operator.add`` accumulate across cycles
    (used for rewrite counters). All other fields are replaced on update.
    """

    # ── Input ──────────────────────────────────────────────────────────────────
    request_id: str
    query: str
    ts_start: float

    # ── Validation ─────────────────────────────────────────────────────────────
    validation_error: str | None

    # ── Retrieval ──────────────────────────────────────────────────────────────
    effective_query: str           # original or most-recent rewrite
    retrieved_hits: list[RetrievalHit]
    graded_hits: list[RetrievalHit]  # after threshold + dedup

    # ── Cycle counter (accumulates via operator.add across rewrite cycles) ──────
    rewrite_count: Annotated[int, operator.add]

    # ── Query rewriting ────────────────────────────────────────────────────────
    rewritten_query: str | None

    # ── Generation ─────────────────────────────────────────────────────────────
    answer_stream: Iterator[str] | None
    answer_text: str | None          # used for abstain / static messages
    provider_used: str | None
    fallback_reason: str | None

    # ── Final output ───────────────────────────────────────────────────────────
    sources: list[RetrievalHit]
    status: str                      # success | abstained | validation_error
                                     # | retrieval_failure | provider_failure

    error_message: str | None

    # ── Telemetry ──────────────────────────────────────────────────────────────
    retrieval_ms: float | None
    generation_ms: float | None
