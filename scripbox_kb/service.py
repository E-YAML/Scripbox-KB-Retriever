"""
Service orchestration module for the Knowledge Base Retriever.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from scripbox_kb import prompting, providers
from scripbox_kb.config import AppConfig
from scripbox_kb.kb_store import KBError, KBStore, RetrievalHit
from scripbox_kb.providers import (
    ProviderAuthError,
    ProviderError,
    ProviderModelNotFoundError,
    ProviderRateLimitError,
)


@dataclass
class ServiceResponse:
    """Standardized response from the query service."""

    status: (
        str  # "success", "abstained", "retrieval_failure", "provider_failure", "validation_error"
    )
    answer_stream: Iterator[str] | None = None
    answer_text: str | None = None
    sources: list[RetrievalHit] = field(default_factory=list)
    provider_used: str | None = None
    fallback_reason: str | None = None
    error_message: str | None = None


class QueryService:
    """Service to handle user queries against the knowledge base."""

    def __init__(self, config: AppConfig, kb_store: KBStore) -> None:
        """Initialize the QueryService.

        Args:
            config: Application configuration.
            kb_store: The knowledge base store to retrieve documents from.
        """
        self._config = config
        self._kb = kb_store

    def handle_query(self, query: str) -> ServiceResponse:
        """Process a query and return a streaming response or appropriate status.

        Args:
            query: The user's query string.

        Returns:
            A ServiceResponse object containing the result or error status.
        """
        validation_msg = prompting.validate_input(query, self._config.max_input_length)
        if validation_msg is not None:
            return ServiceResponse(
                status="validation_error",
                error_message=validation_msg,
            )

        try:
            hits = self._kb.retrieve(query, self._config.top_k)
        except KBError as e:
            return ServiceResponse(
                status="retrieval_failure",
                error_message=str(e),
            )

        if not hits:
            return ServiceResponse(
                status="abstained",
                answer_text=prompting.INSUFFICIENT_CONTEXT_MESSAGE,
                sources=[],
            )

        hits_as_dicts = [
            {
                "title": hit.title,
                "url": hit.url,
                "category": hit.category,
                "folder": hit.folder,
                "document": hit.document,
                "score": hit.score,
            }
            for hit in hits
        ]
        prompt = prompting.build_prompt(query, hits_as_dicts, self._config.max_context_chars)

        try:
            result = providers.stream_with_fallback(
                prompt=prompt,
                groq_api_key=self._config.groq_api_key,
                gemini_api_key=self._config.gemini_api_key,
                groq_model=self._config.groq_model,
                gemini_model=self._config.gemini_model,
                system_prompt=prompting.SYSTEM_PROMPT,
                max_tokens=self._config.max_output_tokens,
            )
        except ProviderError as e:
            return ServiceResponse(
                status="provider_failure",
                error_message=self._sanitize_error(e),
            )

        return ServiceResponse(
            status="success",
            answer_stream=result.stream,
            sources=hits,
            provider_used=result.provider,
            fallback_reason=result.fallback_reason,
        )

    def _sanitize_error(self, exc: Exception) -> str:
        """Sanitize exception messages for user-facing output.

        Args:
            exc: The exception to sanitize.

        Returns:
            A user-safe error message string.
        """
        if isinstance(exc, ProviderRateLimitError):
            return "Our AI providers are temporarily busy. Please try again in a moment."
        elif isinstance(exc, ProviderAuthError):
            return "There's a configuration issue with our AI service. Please contact support."
        elif isinstance(exc, ProviderModelNotFoundError):
            return "The AI model is temporarily unavailable. Please try again later."
        else:
            return "An unexpected error occurred. Please try again."
