import unittest
from unittest.mock import MagicMock, patch

from scripbox_kb import prompting
from scripbox_kb.config import AppConfig
from scripbox_kb.kb_store import KBStore, KBUnavailableError, RetrievalHit
from scripbox_kb.providers import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderResult,
)
from scripbox_kb.service import QueryService


class TestQueryService(unittest.TestCase):
    def setUp(self):
        self.mock_config = AppConfig(
            groq_api_key="test-key",
            gemini_api_key="test-key",
            groq_model="test-model",
            gemini_model="test-model",
            top_k=5,
            max_input_length=2000,
            max_context_chars=1800,
            max_output_tokens=1024,
        )
        self.mock_kb = MagicMock(spec=KBStore)
        self.service = QueryService(self.mock_config, self.mock_kb)

    @patch("scripbox_kb.prompting.validate_input")
    def test_handle_query_empty(self, mock_validate):
        mock_validate.return_value = "Query cannot be empty."
        response = self.service.handle_query("")
        self.assertEqual(response.status, "validation_error")
        self.assertEqual(response.error_message, "Query cannot be empty.")

    @patch("scripbox_kb.prompting.validate_input")
    def test_handle_query_too_long(self, mock_validate):
        mock_validate.return_value = "Query is too long."
        response = self.service.handle_query("A" * 3000)
        self.assertEqual(response.status, "validation_error")
        self.assertEqual(response.error_message, "Query is too long.")

    def test_handle_query_kb_unavailable(self):
        self.mock_kb.retrieve.side_effect = KBUnavailableError("KB is down")
        response = self.service.handle_query("test query")
        self.assertEqual(response.status, "retrieval_failure")
        self.assertEqual(response.error_message, "KB is down")

    def test_handle_query_no_hits(self):
        self.mock_kb.retrieve.return_value = []
        response = self.service.handle_query("test query")
        self.assertEqual(response.status, "abstained")
        self.assertEqual(response.answer_text, prompting.INSUFFICIENT_CONTEXT_MESSAGE)
        self.assertEqual(response.sources, [])

    @patch("scripbox_kb.providers.stream_with_fallback")
    def test_handle_query_success(self, mock_stream):
        hits = [
            RetrievalHit(
                title="T1",
                url="http://example.com/1",
                category="Cat",
                folder="Folder",
                document="Content 1",
                score=0.9,
            )
        ]
        self.mock_kb.retrieve.return_value = hits
        mock_result = ProviderResult(stream=iter(["hello"]), provider="groq")
        mock_stream.return_value = mock_result

        response = self.service.handle_query("test query")

        self.assertEqual(response.status, "success")
        self.assertEqual(response.sources, hits)
        self.assertEqual(response.provider_used, "groq")
        self.assertIsNotNone(response.answer_stream)

    @patch("scripbox_kb.providers.stream_with_fallback")
    def test_handle_query_provider_error(self, mock_stream):
        hits = [
            RetrievalHit(
                title="T1",
                url="http://example.com/1",
                category="Cat",
                folder="Folder",
                document="Content 1",
                score=0.9,
            )
        ]
        self.mock_kb.retrieve.return_value = hits
        mock_stream.side_effect = ProviderRateLimitError("Rate limit exceeded", "groq")

        response = self.service.handle_query("test query")

        self.assertEqual(response.status, "provider_failure")
        self.assertEqual(
            response.error_message,
            "Our AI providers are temporarily busy. Please try again in a moment.",
        )

    def test_sanitize_error_rate_limit(self):
        msg = self.service._sanitize_error(ProviderRateLimitError("error", "groq"))
        self.assertEqual(
            msg, "Our AI providers are temporarily busy. Please try again in a moment."
        )

    def test_sanitize_error_auth(self):
        msg = self.service._sanitize_error(ProviderAuthError("error", "groq"))
        self.assertEqual(
            msg, "There's a configuration issue with our AI service. Please contact support."
        )

    def test_sanitize_error_generic(self):
        msg = self.service._sanitize_error(Exception("error"))
        self.assertEqual(msg, "An unexpected error occurred. Please try again.")


if __name__ == "__main__":
    unittest.main()
