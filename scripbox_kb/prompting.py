"""Module for constructing LLM prompts."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a helpful, friendly customer support assistant for Scripbox, "
    "an investment platform. Answer the user's question using ONLY the "
    "information from the provided knowledge base articles. Be concise, "
    "accurate, and professional. Use bullet points where appropriate. "
    "If the articles don't contain enough information to fully answer, "
    "say so honestly and suggest the user contacts Scripbox support. "
    "Do NOT list article citations in your answer — they are shown separately in the UI."
)

STARTER_QUESTIONS = [
    "How do I update my bank account details?",
    "What is KYC and how do I complete it?",
    "How do I withdraw my investments?",
    "What happens if my SIP payment fails?",
    "How do I track my portfolio performance?",
    "Can I invest on behalf of my child?",
]

WELCOME_MESSAGE = (
    "\U0001f44b Hello! I'm your Scripbox Help Assistant. "
    "I can answer questions about investing, KYC, withdrawals, account management, "
    "and more — all sourced directly from the official Scripbox Knowledge Base. "
    "What would you like to know?"
)

INSUFFICIENT_CONTEXT_MESSAGE = (
    "I couldn't find enough relevant information in the Scripbox Knowledge Base "
    "to answer your question confidently. Please try rephrasing your question, "
    "or contact [Scripbox Support](https://help.scripbox.com) directly for assistance."
)


def build_prompt(query: str, hits: list[dict], max_doc_chars: int = 1800) -> str:
    """Build the prompt for the LLM using the retrieved knowledge base hits.

    Args:
        query: The user's question.
        hits: A list of dictionaries representing the retrieved documents.
        max_doc_chars: Maximum characters to include for each document.

    Returns:
        The formatted prompt string.
    """
    parts = []
    for i, hit in enumerate(hits, 1):
        clean_doc = hit["document"].replace("\n", " ")[:max_doc_chars]
        parts.append(
            f"[Article {i}: {hit['title']}]\n"
            f"Category: {hit['category']} > {hit['folder']}\n"
            f"URL: {hit['url']}\n\n"
            f"{clean_doc}"
        )
    context = "\n\n---\n\n".join(parts)
    return f"KNOWLEDGE BASE ARTICLES:\n{context}\n\nUSER QUESTION: {query}\n\nANSWER:"


def validate_input(query: str, max_length: int = 2000) -> str | None:
    """Validate the user query.

    Args:
        query: The user's question.
        max_length: Maximum allowed length for the query.

    Returns:
        None if the query is valid, or an error message string if invalid.
    """
    query = query.strip()
    if not query:
        return "Please enter a valid question."
    if len(query) > max_length:
        return f"Your question is too long. Please limit it to {max_length} characters."
    return None
