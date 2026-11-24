"""Node factory functions for the Scripbox cyclic RAG graph.

Each ``make_*`` function closes over ``config`` / ``kb_store`` and returns
a plain callable suitable as a LangGraph node.

Graph topology:
    START
      └─► validate_input ──[error]──────────────────────────► END
                 │ [valid]
                 ▼
              retrieve
                 │
                 ▼
           grade_documents ──[error]──────────────────────► END
                 │                        ▲
         [sufficient]              [rewrite] ← cycle (max N)
                 │                        │
                 ▼                        │
             generate               rewrite_query
                 │                        ▲
                 ▼                        │
               END          [abstain] ────┘
                              (after MAX_REWRITES)
"""
from __future__ import annotations

import time
from collections.abc import Callable

from scripbox_kb import observability, prompting, providers
from scripbox_kb.config import AppConfig
from scripbox_kb.graph_state import RAGState
from scripbox_kb.kb_store import KBStore, RetrievalHit
from scripbox_kb.prompting import INSUFFICIENT_CONTEXT_MESSAGE


# ─── Node factories ───────────────────────────────────────────────────────────


def make_validate_node(config: AppConfig) -> Callable[[RAGState], dict]:
    """Return a node that validates user input and filters out-of-scope requests."""

    def validate_input(state: RAGState) -> dict:
        query = state["query"]
        error = prompting.validate_input(query, config.max_input_length)
        if error:
            return {
                "validation_error": error,
                "status": "validation_error",
                "error_message": error,
            }

        # Fast pre-retrieval guardrail for creative or stock-tipping requests
        q_lower = query.lower()
        out_of_scope_patterns = [
            "poem", "poetry", "write a song", "write me a poem", "tell me a joke",
            "write a story", "best stock to buy", "best stocks to buy",
            "which stock to buy", "which stock should i buy", "stock tip",
        ]
        if any(p in q_lower for p in out_of_scope_patterns):
            return {
                "status": "abstained",
                "answer_text": INSUFFICIENT_CONTEXT_MESSAGE,
                "sources": [],
            }

        return {
            "validation_error": None,
            "effective_query": query,
        }

    return validate_input


def make_retrieve_node(config: AppConfig, kb_store: KBStore) -> Callable[[RAGState], dict]:
    """Return a node that retrieves candidate chunks from the vector DB."""

    def retrieve(state: RAGState) -> dict:
        # Use the most recent (possibly rewritten) query
        query = state.get("effective_query") or state["query"]
        t0 = time.perf_counter()

        try:
            hits = kb_store.retrieve(query, top_k=config.retrieval_candidates)
        except Exception as exc:
            return {
                "retrieved_hits": [],
                "graded_hits": [],
                "status": "retrieval_failure",
                "error_message": str(exc),
                "retrieval_ms": (time.perf_counter() - t0) * 1000,
            }

        return {
            "retrieved_hits": hits,
            "retrieval_ms": (time.perf_counter() - t0) * 1000,
        }

    return retrieve


def make_grade_documents_node(config: AppConfig) -> Callable[[RAGState], dict]:
    """Return a node that filters candidates by similarity threshold and deduplicates by URL."""

    def grade_documents(state: RAGState) -> dict:
        hits = state.get("retrieved_hits", [])
        threshold = config.retrieval_threshold

        # 1. Threshold filter
        relevant = [h for h in hits if h.score >= threshold]

        # 2. Deduplicate by article title & URL — keep highest-scoring chunk per article
        seen: dict[str, RetrievalHit] = {}
        for h in relevant:
            key = h.title.strip().lower()
            if key not in seen or h.score > seen[key].score:
                seen[key] = h

        graded = sorted(seen.values(), key=lambda x: x.score, reverse=True)[: config.top_k]

        top_score = hits[0].score if hits else None
        rewrite_count = state.get("rewrite_count", 0)

        observability.log_retrieval(
            request_id=state["request_id"],
            duration_ms=state.get("retrieval_ms") or 0.0,
            candidate_count=len(hits),
            relevant_count=len(graded),
            top_score=top_score,
            confidence_decision="sufficient" if graded else "insufficient",
            rewrite_count=rewrite_count,
        )

        return {"graded_hits": graded}

    return grade_documents


def make_rewrite_node(config: AppConfig) -> Callable[[RAGState], dict]:
    """Return a node that rewrites the query via LLM for better retrieval coverage."""
    import re

    def rewrite_query(state: RAGState) -> dict:
        original = state["query"]
        attempt = state.get("rewrite_count", 0) + 1

        observability.log_rewrite(request_id=state["request_id"], attempt=attempt)

        rewrite_prompt = prompting.build_rewrite_prompt(original, attempt)

        raw = providers.call_once(
            prompt=rewrite_prompt,
            groq_api_key=config.groq_api_key,
            gemini_api_key=config.gemini_api_key,
            groq_model=config.groq_model,
            gemini_model=config.gemini_model,
            system_prompt=prompting.QUERY_REWRITER_SYSTEM,
            max_tokens=512,
        ).strip()

        # Clean any introductory headings or markdown quotes
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        candidate = lines[-1] if lines else raw
        cleaned = re.sub(r'[*"`]|^(Rewritten (search )?query:?|\bQuery:?)\s*', '', candidate, flags=re.I).strip()

        # If rewriter identified the query as out-of-domain, force abstention
        if cleaned.upper() == "UNRELATED" or raw.strip().upper() == "UNRELATED":
            return {
                "rewritten_query": "UNRELATED",
                "effective_query": "UNRELATED",
                "rewrite_count": 999,  # exceeds max_rewrites -> routes to abstain
            }

        effective = cleaned or original

        return {
            "rewritten_query": effective,
            "effective_query": effective,
            "rewrite_count": 1,  # accumulates via Annotated[int, operator.add]
        }

    return rewrite_query


def make_abstain_node() -> Callable[[RAGState], dict]:
    """Return a node that marks the response as abstained (no relevant docs found)."""

    def abstain(state: RAGState) -> dict:
        return {
            "status": "abstained",
            "answer_text": INSUFFICIENT_CONTEXT_MESSAGE,
            "sources": [],
        }

    return abstain


def make_generate_node(config: AppConfig) -> Callable[[RAGState], dict]:
    """Return a node that calls the LLM and stores a streaming iterator in state."""

    def generate(state: RAGState) -> dict:
        # Always generate against the user's original question so the answer
        # speaks directly to the user's inquiry rather than search-keyword phrases.
        query = state["query"]
        hits = state.get("graded_hits", [])

        hits_as_dicts = [
            {
                "title": h.title,
                "url": h.url,
                "category": h.category,
                "folder": h.folder,
                "document": h.document,
                "score": h.score,
            }
            for h in hits
        ]
        prompt = prompting.build_prompt(query, hits_as_dicts, config.max_context_chars)

        t0 = time.perf_counter()
        try:
            result = providers.stream_with_fallback(
                prompt=prompt,
                groq_api_key=config.groq_api_key,
                gemini_api_key=config.gemini_api_key,
                groq_model=config.groq_model,
                gemini_model=config.gemini_model,
                system_prompt=prompting.SYSTEM_PROMPT,
                max_tokens=config.max_output_tokens,
            )
            generation_ms = (time.perf_counter() - t0) * 1000

            observability.log_generation(
                request_id=state["request_id"],
                provider=result.provider,
                fallback_reason=result.fallback_reason,
                ttft_ms=None,  # first token already pre-fetched in stream_with_fallback
                total_ms=generation_ms,
            )

            return {
                "answer_stream": result.stream,
                "provider_used": result.provider,
                "fallback_reason": result.fallback_reason,
                "sources": hits,
                "status": "success",
                "generation_ms": generation_ms,
            }

        except providers.ProviderError as exc:
            generation_ms = (time.perf_counter() - t0) * 1000
            observability.log_generation(
                request_id=state["request_id"],
                provider="unknown",
                fallback_reason=None,
                ttft_ms=None,
                total_ms=generation_ms,
                error_class=type(exc).__name__,
            )
            return {
                "status": "provider_failure",
                "error_message": providers.sanitize_error(exc),
                "generation_ms": generation_ms,
            }

    return generate


# ─── Routing functions ────────────────────────────────────────────────────────


def route_after_validation(state: RAGState) -> str:
    """Route: error -> END, abstain -> END, valid -> retrieve."""
    if state.get("validation_error") or state.get("status") == "validation_error":
        return "error"
    if state.get("status") == "abstained":
        return "abstain"
    return "valid"


def route_after_grading(state: RAGState) -> str:
    """Route based on document grading outcome.

    - retrieval_failure  → error (END)
    - graded docs found  → sufficient (generate)
    - rewrite budget left → rewrite (cycle back to retrieve)
    - budget exhausted   → abstain (END)
    """
    if state.get("status") == "retrieval_failure":
        return "error"

    graded = state.get("graded_hits", [])
    rewrite_count = state.get("rewrite_count", 0)

    if graded:
        return "sufficient"

    if rewrite_count < _MAX_REWRITES:
        return "rewrite"

    return "abstain"


# Module-level constant used by routing (overridden at graph build time)
_MAX_REWRITES: int = 2


def set_max_rewrites(n: int) -> None:
    """Configure the global max-rewrite limit (called from graph.py)."""
    global _MAX_REWRITES
    _MAX_REWRITES = n
