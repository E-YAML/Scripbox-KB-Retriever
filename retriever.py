"""
retriever.py — Scripbox KB Interactive Q&A
Ask a question and get an answer sourced from Scripbox help articles.

LLM priority (first available key/service wins):
  1. Groq  — free API, fast Llama 3 (set GROQ_API_KEY in .env)
  2. Ollama — fully local, no key needed (install from ollama.com)
  3. Gemini — fallback if GEMINI_API_KEY is set

Usage:
    python retriever.py
    python retriever.py "How do I reset my password?"

Requires:
    - chroma_db/ directory (run build_index.py first)
"""
from __future__ import annotations  # Python 3.8+ compatible type hints

import sys
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

# ── ANSI color support (Windows + macOS/Linux) ────────────────────────────────
try:
    from colorama import init as _colorama_init
    _colorama_init(autoreset=False)   # enables ANSI on Windows terminals
except ImportError:
    pass   # colorama optional; ANSI works natively on macOS/Linux

# ── Load env vars ─────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_HOST    = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "llama3")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
import chromadb

# ── Sentence Transformers ─────────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer

CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "scripbox_kb"
EMBED_MODEL     = "all-MiniLM-L6-v2"
TOP_K           = 5

# ANSI colors
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"


# ══════════════════════════════════════════════════════════════════════════════
# LLM backends
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant for Scripbox, an investment platform. "
    "Answer the user's question using ONLY the information from the provided knowledge base articles. "
    "If the articles don't contain enough information, say so honestly. "
    "Always cite which article(s) you used at the end of your answer."
)


def _build_prompt(query: str, hits: list) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        parts.append(
            f"[Article {i}: {hit['title']}]\n"
            f"Category: {hit['category']} > {hit['folder']}\n"
            f"URL: {hit['url']}\n\n"
            f"{hit['document'][:1500]}"
        )
    context = "\n\n---\n\n".join(parts)
    return (
        f"KNOWLEDGE BASE ARTICLES:\n{context}\n\n"
        f"USER QUESTION: {query}\n\nANSWER:"
    )


def _synthesize_groq(query: str, hits: list) -> str:
    """Groq free API — Llama 3 70B, 14,400 req/day free tier."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    prompt = _build_prompt(query, hits)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err = str(e)
            if "429" in err and attempt < 2:
                m = re.search(r"try again in ([\d.]+)s", err, re.I)
                wait = float(m.group(1)) + 1 if m else 30
                print(f"  {YELLOW}Groq rate limited — waiting {int(wait)}s...{RESET}")
                time.sleep(wait)
                continue
            return f"[Groq error: {e}]"
    return "[Groq error: max retries exceeded]"


def _synthesize_ollama(query: str, hits: list) -> str:
    """Ollama — runs a local LLM, completely free, no rate limits."""
    import json
    import urllib.request
    prompt = _build_prompt(query, hits)
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
            return data.get("response", "[Ollama: empty response]")
    except Exception as e:
        return f"[Ollama error: {e}]"


def _synthesize_gemini(query: str, hits: list) -> str:
    """Gemini flash — fallback."""
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = _build_prompt(query, hits)
    for attempt in range(3):
        try:
            return model.generate_content(f"{SYSTEM_PROMPT}\n\n{prompt}").text
        except Exception as e:
            err = str(e)
            m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", err)
            wait = int(m.group(1)) + 2 if m else None
            if "429" in err and wait and attempt < 2:
                print(f"  {YELLOW}Gemini rate limited — waiting {wait}s...{RESET}")
                time.sleep(wait)
                continue
            return f"[Gemini error: {e}]"
    return "[Gemini error: max retries exceeded]"


def _ollama_running() -> bool:
    """Check whether the Ollama server is reachable."""
    import urllib.request
    try:
        urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return True
    except Exception:
        return False


def detect_llm_backend() -> tuple:
    """
    Auto-detect which LLM to use.
    Returns (backend_name: str, description: str).
    Priority: Groq > Ollama > Gemini > None
    """
    if GROQ_API_KEY:
        return ("groq", "Groq (llama-3.3-70b-versatile) — free tier")
    if _ollama_running():
        return ("ollama", f"Ollama local ({OLLAMA_MODEL})")
    if GEMINI_API_KEY:
        return ("gemini", "Gemini 2.0 Flash")
    return ("none", "No LLM configured — showing raw snippets only")


def synthesize(backend: str, query: str, hits: list) -> "str | None":
    if backend == "groq":
        return _synthesize_groq(query, hits)
    if backend == "ollama":
        return _synthesize_ollama(query, hits)
    if backend == "gemini":
        return _synthesize_gemini(query, hits)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ChromaDB + retrieval
# ══════════════════════════════════════════════════════════════════════════════

def load_collection():
    db_path = Path(CHROMA_DIR)
    if not db_path.exists():
        print(f"{YELLOW}ERROR: '{CHROMA_DIR}' not found. Run build_index.py first.{RESET}")
        sys.exit(1)
    client = chromadb.PersistentClient(path=str(db_path))
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"{YELLOW}ERROR: Collection '{COLLECTION_NAME}' not found. Run build_index.py first.{RESET}")
        sys.exit(1)


def retrieve(query: str, embed_model: SentenceTransformer, collection) -> list:
    qv = embed_model.encode(query).tolist()
    res = collection.query(
        query_embeddings=[qv],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id":       res["ids"][0][i],
            "title":    res["metadatas"][0][i].get("title", ""),
            "url":      res["metadatas"][0][i].get("url", ""),
            "category": res["metadatas"][0][i].get("category", ""),
            "folder":   res["metadatas"][0][i].get("folder", ""),
            "document": res["documents"][0][i],
            "score":    1 - res["distances"][0][i],
        })
    return hits


# ══════════════════════════════════════════════════════════════════════════════
# Output
# ══════════════════════════════════════════════════════════════════════════════

def print_results(query: str, hits: list, answer: "str | None"):
    print()
    print(f"{BOLD}{CYAN}{'=' * 70}{RESET}")
    print(f"{BOLD}  Question: {query}{RESET}")
    print(f"{CYAN}{'=' * 70}{RESET}")

    if answer:
        print(f"\n{BOLD}{GREEN}AI Answer:{RESET}")
        print(answer)
        print()

    print(f"{BOLD}Source Articles (Top {len(hits)}):{RESET}")
    print(f"{DIM}{'-' * 70}{RESET}")
    for i, hit in enumerate(hits, 1):
        score_pct = int(hit["score"] * 100)
        print(f"\n{BOLD}  [{i}] {hit['title']}{RESET}")
        print(f"      {DIM}Category:{RESET} {hit['category']} > {hit['folder']}")
        print(f"      {DIM}Relevance:{RESET} {score_pct}%")
        print(f"      {DIM}URL:{RESET} {CYAN}{hit['url']}{RESET}")
        snippet = hit["document"].replace("Title: ", "").replace("\n", " ").strip()
        lines   = [ln for ln in snippet.split("  ") if ln.strip()]
        preview = " ".join(lines)[:250] + "..." if len(snippet) > 250 else snippet
        print(f"      {DIM}Preview:{RESET} {preview}")

    print(f"\n{DIM}{'-' * 70}{RESET}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Verify Python version
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required.")
        sys.exit(1)

    single_query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""

    print("Loading vector store and embedding model...")
    collection  = load_collection()
    embed_model = SentenceTransformer(EMBED_MODEL)

    backend, backend_desc = detect_llm_backend()
    print(f"Ready  |  LLM: {backend_desc}")

    if single_query:
        hits   = retrieve(single_query, embed_model, collection)
        answer = synthesize(backend, single_query, hits)
        print_results(single_query, hits, answer)
        return

    # ── Interactive loop ──────────────────────────────────────────────────────
    print(f"\n{BOLD}{GREEN}Scripbox Knowledge Base Retriever{RESET}")
    print(f"   Indexed articles : {collection.count()}")
    print(f"   LLM backend      : {backend_desc}")
    if backend == "none":
        print(f"   {DIM}Tip: Add GROQ_API_KEY to .env for free AI answers (console.groq.com){RESET}")
        print(f"   {DIM}  or install Ollama (ollama.com) and run: ollama pull llama3{RESET}")
    print(f"\n   Type your question and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            query = input(f"{BOLD}{YELLOW}Ask: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print(f"\n{DIM}Searching...{RESET}")
        hits = retrieve(query, embed_model, collection)

        answer = None
        if backend != "none":
            print(f"{DIM}Synthesizing with {backend_desc}...{RESET}")
            answer = synthesize(backend, query, hits)

        print_results(query, hits, answer)


if __name__ == "__main__":
    main()
