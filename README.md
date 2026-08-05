# Scripbox Knowledge Base Assistant

A production-grade RAG (Retrieval-Augmented Generation) tool that searches the [Scripbox Help Center](https://help.scripbox.com/support/solutions) (~211 articles) and answers questions with Groq Llama 3.3-70b — with streaming responses, source citations, and a polished Streamlit UI.

## Live Demo

> Open the app → see real-time streaming answers sourced from the official Scripbox Knowledge Base.

---

## Quick Start

```bash
# 1. Clone
git clone <your-repo-url>
cd scripbox-kb-retriever

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\Activate.ps1         # Windows PowerShell

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key (free at console.groq.com)
# Edit .streamlit/secrets.toml:
#   GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"

# 5. Build the knowledge base (run ONCE)
python scraper.py       # scrape articles (~3 min)
python build_index.py   # build vector index (~30 sec)

# 6. Launch the app
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## Setting Up the Groq API Key

Everyone who uses your hosted app shares **one Groq API key** — you set it once and it's used for all users.

1. Get a free key at [console.groq.com](https://console.groq.com) → **API Keys** → **Create**
2. Open `.streamlit/secrets.toml` (already created for you) and paste it:

```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxx"
```

> **Security**: `secrets.toml` is in `.gitignore` — it will never be committed.

---

## Architecture

```
help.scripbox.com
       │
  scraper.py  ──────────────→  articles.json  (211 articles)
                                      │
                               build_index.py
                              (sentence-transformers
                               all-MiniLM-L6-v2)
                                      │
                                  chroma_db/   (vector store)
                                      │
                  ┌───────────────────┘
            User question
                  │
           Embed query (all-MiniLM-L6-v2)
                  │
           ChromaDB top-5 search
                  │
           Groq Llama 3.3-70b (streaming)
                  │
         Streamed answer + source citations
```

---

## Features

| Feature | Detail |
|---|---|
| **Streaming answers** | Groq Llama 3.3-70b, token-by-token output |
| **Source citations** | Top-5 matching articles with relevance % |
| **Starter prompts** | 6 curated example questions to click |
| **KB stats** | Article count + category browser in sidebar |
| **System status** | Live DB + LLM health indicators |
| **Shared API key** | One Groq key for all users — set in secrets.toml |
| **Robust errors** | Friendly messages for rate limits, auth errors, missing DB |

---

## Keeping the Index Up to Date

```bash
# Full re-scrape (when Scripbox adds many new articles)
python scraper.py
python build_index.py

# Incremental patch (faster — only adds missing articles)
python patch_articles.py
```

---

## CLI Usage (without Streamlit)

```bash
# Interactive Q&A in terminal
python retriever.py

# Single-shot query
python retriever.py "How do I withdraw my investments?"
```

---

## File Reference

```
scripbox-kb-retriever/
├── app.py                          # Streamlit web UI (production)
├── retriever.py                    # CLI retriever (Groq / Ollama / Gemini)
├── scraper.py                      # Crawls help.scripbox.com → articles.json
├── build_index.py                  # Embeds articles → chroma_db/
├── patch_articles.py               # Incremental index updater
├── requirements.txt                # Pinned Python dependencies
├── .streamlit/
│   ├── config.toml                 # Streamlit theme (Scripbox brand colors)
│   ├── secrets.toml                # Your Groq key (git-ignored)
│   └── secrets.toml.example        # Key format reference
├── .env.example                    # Env var reference (for CLI use)
├── articles.json                   # Scraped articles (git-ignored)
└── chroma_db/                      # Vector store (git-ignored)
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `chroma_db/ not found` | Run `python build_index.py` first |
| `articles.json not found` | Run `python scraper.py` first |
| `Groq key missing` warning | Add key to `.streamlit/secrets.toml` |
| `401 Unauthorized` | Key is invalid — check console.groq.com |
| `429 Rate limit` | Groq free tier hit — wait a moment and retry |
| `Answers seem wrong` | Run `python patch_articles.py` to add any missing articles |
