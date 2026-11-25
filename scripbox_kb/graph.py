"""LangGraph StateGraph builder for the Scripbox cyclic RAG pipeline.

The compiled graph is the core orchestration engine replacing the linear
QueryService.handle_query() call chain.

Cycle structure
---------------
    START ──► validate_input
                  │ [error] ──────────────────────────────► END
                  │ [valid]
                  ▼
               retrieve ◄─────────────────────────────────┐
                  │                                        │
                  ▼                                        │
           grade_documents                           rewrite_query
                  │ [error] ──────────────────────────► END│
                  │ [rewrite] ─────────────────────────────┘
                  │ [abstain] ──────────────────────────► END
                  │ [sufficient]
                  ▼
              generate ──────────────────────────────────► END
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from scripbox_kb.config import AppConfig
from scripbox_kb.graph_nodes import (
    make_abstain_node,
    make_generate_node,
    make_grade_documents_node,
    make_retrieve_node,
    make_rewrite_node,
    make_validate_node,
    route_after_grading,
    route_after_validation,
    set_max_rewrites,
)
from scripbox_kb.graph_state import RAGState
from scripbox_kb.kb_store import KBStore


def build_rag_graph(config: AppConfig, kb_store: KBStore) -> Any:
    """Build and compile the cyclic RAG LangGraph.

    Args:
        config: Application configuration.
        kb_store: Loaded knowledge-base store.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready for ``.invoke()``.
    """
    # Propagate max_rewrites to the routing function
    set_max_rewrites(config.max_rewrites)

    workflow: Any = StateGraph(RAGState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    workflow.add_node("validate_input", make_validate_node(config))
    workflow.add_node("retrieve", make_retrieve_node(config, kb_store))
    workflow.add_node("grade_documents", make_grade_documents_node(config))
    workflow.add_node("rewrite_query", make_rewrite_node(config))
    workflow.add_node("abstain", make_abstain_node())
    workflow.add_node("generate", make_generate_node(config))

    # ── Edges ─────────────────────────────────────────────────────────────────
    workflow.add_edge(START, "validate_input")

    workflow.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {"error": END, "abstain": END, "valid": "retrieve"},
    )

    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "sufficient": "generate",
            "rewrite": "rewrite_query",  # ◄── CYCLE entry
            "abstain": "abstain",
            "error": END,
        },
    )

    # ◄── CYCLE: rewrite sends back to retrieve
    workflow.add_edge("rewrite_query", "retrieve")

    workflow.add_edge("abstain", END)
    workflow.add_edge("generate", END)

    return workflow.compile()
