# Scripbox Knowledge Base Assistant (Production V2)

A production-grade, cyclic RAG (Retrieval-Augmented Generation) assistant that searches the [Scripbox Help Center](https://help.scripbox.com/support/solutions) with LangGraph graph orchestration, semantic chunking, Qdrant/ChromaDB vector stores, structured privacy-safe observability, and an automated evaluation suite.

---

## Architecture: LangGraph Cyclic RAG

The query pipeline runs on a compiled cyclic state machine built with **LangGraph**:

```
START ──► validate_input
              │ [error] ──────────────────────────────► END
              │ [valid]
              ▼
           retrieve ◄─────────────────────────────────┐
              │                                        │
              ▼                                        │
       grade_documents                           rewrite_query
              │ [error] ──────────────────────────► END│
              │ [rewrite] ─────────────────────────────┘  (cycles up to 2×)
              │ [abstain] ──────────────────────────► END  (insufficient context)
              │ [sufficient]
              ▼
          generate ──────────────────────────────────► END
```

1. **Input Validation**: Sanitizes and enforces safety/length bounds on queries.
2. **Retrieval**: Over-fetches candidate chunks (default `top_k=20`) from the vector database.
3. **Grading & Abstention**:
   - Applies cosine similarity threshold (default `0.35`).
   - Deduplicates multiple chunks belonging to the same source article.
   - If no chunks pass the threshold, routes to **Query Rewriter** instead of guessing.
4. **Query Rewriting (Cycle)**:
   - Uses an LLM to reformulate the query with Scripbox domain keywords (KYC, SIP, folio, NAV, ELSS).
   - Cycles back to **retrieve** up to `MAX_REWRITES` (default: 2).
   - If still unsupported after max rewrites, deterministically **abstains** with a support escalation link.
5. **Generation**:
   - Streams completions using **Groq** (`openai/gpt-oss-120b`).
   - Automatically fails over to **Google Gemini** (`gemini-2.5-flash`) on rate-limit or outage.

---

## Free-Tier Vector DB Support: Qdrant Cloud & ChromaDB

The application seamlessly supports both local and hosted vector backends via `VECTOR_DB_TYPE`:

| Vector DB | Mode | Setup | Ideal For |
|---|---|---|---|
| **ChromaDB** (Default) | Local file-based (`./chroma_db`) | Out-of-the-box, zero configuration | Local development & offline runs |
| **Qdrant Cloud** | Managed SaaS (Free Tier: 1GB cluster forever) | Set `QDRANT_URL` and `QDRANT_API_KEY` | Production Streamlit Cloud deployments |

### Configuring Qdrant Cloud Free Tier:
1. Create a free account at [cloud.qdrant.io](https://cloud.qdrant.io).
2. Create a free 1GB cluster (no credit card required).
3. Add your credentials to `.streamlit/secrets.toml`:
```toml
VECTOR_DB_TYPE = "qdrant"
QDRANT_URL     = "https://your-cluster-url.qdrant.io:6333"
QDRANT_API_KEY = "your-api-key"
```
4. Build the chunked vector index directly to Qdrant:
```bash
VECTOR_DB_TYPE=qdrant QDRANT_URL=... QDRANT_API_KEY=... python build_index.py
```

---

## Semantic Chunking

Rather than indexing entire raw articles (which causes noisy matches on long articles), `scripbox_kb.chunker` splits articles into overlapping semantic chunks:
- **Chunk size**: ~400 words (~520 tokens)
- **Overlap**: 80 words
- **Metadata**: Every chunk preserves article `title`, `url`, `category`, `folder`, `section_header`, `chunk_position`, `chunk_total`, and `content_hash`.

---

## Evaluation Suite (`eval/`)

Automated evaluation fixtures and regression runner to guarantee answer and retrieval quality before releases:

```bash
# Run all evaluation fixtures
python -m eval.runner

# Run specific fixture suite
python -m eval.runner --fixtures kyc
python -m eval.runner --fixtures abstention

# Override retrieval threshold
python -m eval.runner --threshold 0.35
```

### Metrics & Quality Gates:
- **Retrieval Recall@5**: Verifies that the correct help article appears in the top-5 retrieved chunks for in-domain queries. Gate: `≥ 80%`.
- **Abstention Rate**: Verifies that out-of-domain, irrelevant, or prompt-injection queries are declined with the support escalation message rather than hallucinating. Gate: `100%`.
- **Structured Reports**: Timestamped results saved to `eval/results/eval_<timestamp>.json`.

---

## Privacy-Safe Structured Observability

Structured JSON telemetry is emitted to `stderr` on every lifecycle event (compatible with log aggregators like Datadog, Grafana Loki, or Streamlit Cloud logs):
- `app_startup`: Configured providers, vector DB backend, embedding model.
- `retrieval`: Request ID, search duration (ms), candidate count, relevant count, top similarity score, confidence decision, rewrite count.
- `query_rewrite`: Request ID and rewrite cycle attempt.
- `generation`: Request ID, provider selected, fallback reason, TTFT (ms), total duration, error classes.
- `response`: Final status (`success`, `abstained`, `validation_error`, `provider_failure`), total latency.
- **Privacy Guarantee**: Raw queries, prompt texts, and completion bodies are **never** logged.

---

## Quick Start

```bash
# 1. Virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# 2. Add keys to .streamlit/secrets.toml
#   GROQ_API_KEY = "gsk_..."
#   GEMINI_API_KEY = "AIza..."

# 3. Build the chunked vector index
python build_index.py

# 4. Run tests
pytest

# 5. Run evaluation suite
python -m eval.runner

# 6. Launch Streamlit UI
streamlit run app.py
```

---

## Testing

```bash
# Run all 61 unit and integration tests
pytest -v
```
All tests verify:
- Semantic chunker boundaries, headers, and metadata preservation.
- Configuration loading, secret resolution, and validation.
- Vector store loading and retrieval across backends.
- Prompt templates and input validation bounds.
- LLM provider error classification and streaming fallbacks.
- LangGraph cyclic routing, query rewrites, and abstention policies.
- Structured observability events.
- Ingestion staging, diffing, atomic promotion, and rollback.
- QueryService concurrency semaphore bounds and timeouts.

---

## Staged Ingestion & Atomic Rollback CLI (`ingest.py`)

```bash
# 1. Stage new articles into a timestamped release directory
python ingest.py crawl [--source articles.json]

# 2. Diff staged release against the currently active release
python ingest.py diff

# 3. Build isolated vector index for the staged release
python ingest.py build --backend chroma   # or --backend qdrant

# 4. Smoke-test and atomically promote staged release to active
python ingest.py promote

# 5. Instant rollback to previous release
python ingest.py rollback

# 6. Show release registry status
python ingest.py status
```

---

## Docker Containerization

Run the application as an isolated container:

```bash
# Using Docker Compose:
docker compose up --build -d

# Or with Docker CLI:
docker build -t scripbox-kb-retriever:v2 .
docker run -p 8501:8501 --env-file .env scripbox-kb-retriever:v2
```

---

## Streamlit Cloud Deployment

1. Push your code to GitHub (branch `v2` or `main`).
2. Log into [share.streamlit.io](https://share.streamlit.io/) and click **New app**.
3. Select your repository, branch, and set Main file path to `app.py`.
4. Under **Advanced Settings** -> **Secrets**, paste your configuration:
   ```toml
   VECTOR_DB_TYPE = "qdrant"
   QDRANT_URL     = "https://your-cluster-url.qdrant.io"
   QDRANT_API_KEY = "your-qdrant-api-key"
   GROQ_API_KEY   = "gsk_..."
   GEMINI_API_KEY = "AIza..." # optional fallback
   ```
5. Click **Deploy!** Since vector indices are hosted in Qdrant Cloud, the deployment is fully stateless and persists across server reboots.
