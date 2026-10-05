# Scripbox KB Retriever — V2 Plan

## Purpose

Evolve the current Streamlit RAG prototype into a dependable small-user support tool with fresher knowledge, more grounded answers, predictable failure behaviour, and a maintainable delivery process.

This plan is intentionally scoped for the current **two-user** deployment. User accounts, per-user quotas, public abuse protection, and provider spend controls are deferred for now.

## V2 outcomes

By the end of V2, the system should:

1. Refresh the Scripbox knowledge base repeatably and publish an index only after validation.
2. Prefer grounded, cited answers and decline to guess when retrieval confidence is weak.
3. Continue serving useful errors when the index or an LLM provider is unavailable.
4. Produce enough telemetry to diagnose failures and judge answer quality without logging sensitive user content unnecessarily.
5. Have automated tests and repeatable builds so changes can be released with confidence.

## Current baseline

```text
Streamlit UI
  → all-MiniLM-L6-v2 query embedding
  → local ChromaDB top-5 full-article retrieval
  → Groq streaming response
  → Gemini streaming fallback
  → rendered source articles
```

The live application works for the existing small audience. V2 retains Streamlit as the user interface, but separates business logic from UI code so the serving layer can evolve later without a rewrite.

## Scope and non-goals

### In scope

- Versioned, validated knowledge-base ingestion.
- Chunk-level retrieval, retrieval thresholds, and answer abstention.
- Evaluation fixtures and automated retrieval/answer-quality checks.
- Provider resiliency, structured logs, health checks, and sanitized errors.
- Configuration validation, dependency locking, tests, and CI.

### Deferred

- Login, authentication, and identity management.
- Per-user quotas, billing, spend budgets, and public bot protection.
- Multi-region or high-availability infrastructure.
- A full frontend rewrite or replacing Streamlit.
- Human-agent ticket creation/integration.

### Low-cost safeguards retained

Although full request controls are deferred, V2 will retain:

- A maximum input length.
- A maximum retrieved-context size.
- A maximum output-token limit.
- A small in-process request concurrency limit to avoid accidental overload.

These are reliability limits, not a user-quota system.

## Architecture target

```text
                         ┌──────────────────────────────────┐
Scripbox Help Center ──► │ Ingestion job                     │
                         │ crawl → normalize → diff → validate│
                         └──────────────┬───────────────────┘
                                        │
                         ┌──────────────▼───────────────────┐
                         │ Versioned KB release              │
                         │ documents + metadata + index      │
                         └──────────────┬───────────────────┘
                                        │
User ─► Streamlit UI ─► Application service
                         │ input validation
                         │ retrieval + confidence decision
                         │ generation + fallback
                         │ citations + safe errors
                         ▼
                    Groq / Gemini
```

## Workstreams

### 1. Application-service refactor

**Goal:** Move non-UI logic out of `app.py` without changing the visible experience.

Create modules for:

- `config.py`: typed environment/secrets configuration and startup validation.
- `kb_store.py`: resource loading, retrieval, and index metadata checks.
- `prompting.py`: system prompt, prompt construction, and response policy.
- `providers.py`: Groq/Gemini streaming clients, typed provider failures, and fallback policy.
- `service.py`: request validation, orchestration, confidence decisions, and response model.
- `observability.py`: privacy-safe structured event logging and timing helpers.

`app.py` should only render Streamlit UI and call the service layer.

**Acceptance criteria**

- Existing starter questions still work end to end.
- Provider selection and fallback are covered by automated tests.
- Startup identifies missing/mismatched KB release files clearly.
- UI never exposes raw exception text or secret-derived details.

### 2. Knowledge ingestion and release management

**Goal:** Make KB updates reproducible, reviewable, and reversible.

Implement a release directory format, for example:

```text
data/releases/
  2026-10-05T120000Z/
    articles.json
    manifest.json
    index/
  current.json
```

The manifest should record:

- Release ID and generation time.
- Source crawl start/end times.
- Article count, chunk count, and category count.
- Source URLs and content hashes.
- Embedder model and version.
- Index/storage schema version.
- Validation result and previous release ID.

The ingestion workflow should:

1. Crawl into a staging directory.
2. Normalize text and discard empty/duplicate documents.
3. Diff the staged source set against the current release.
4. Build a new chunk-level index in a separate staged directory.
5. Run validation and retrieval regression tests.
6. Atomically promote the staged release by updating `current.json`.
7. Retain at least one previous release for rollback.

**Acceptance criteria**

- A failed crawl/build cannot overwrite the currently served KB.
- Each release can be reproduced from its manifest.
- Rollback changes the active release without rebuilding it.
- A scheduled task can run the same workflow unattended after its first manually reviewed runs.

### 3. Retrieval and grounding quality

**Goal:** Answer from the KB when the KB supports the answer; otherwise abstain.

Changes:

- Split articles into overlapping semantic chunks, retaining article title, URL, category, section, content hash, and chunk position as metadata.
- Retrieve more candidates than are shown to the model, then deduplicate by article.
- Start with configurable top-k and distance thresholds instead of treating `1 - distance` as an absolute relevance percentage.
- Add a deterministic `insufficient_context` decision when no result passes the threshold.
- Require response claims to cite source IDs internally; map those IDs to user-visible links.
- Keep the support escalation path when confidence is insufficient.

Optional later V2 increment:

- Add lexical/BM25 retrieval and reciprocal-rank fusion.
- Add a small cross-encoder reranker if latency remains acceptable for two users.

**Acceptance criteria**

- Out-of-domain and low-evidence questions decline to answer rather than hallucinate.
- Duplicate chunks do not dominate context.
- Every surfaced source resolves to the document used by retrieval.
- The UI labels sources as “Retrieved sources”, not a percentage that implies calibrated factual confidence.

### 4. Evaluation suite

**Goal:** Make retrieval and answer quality measurable before releases.

Create a versioned evaluation set containing:

- Common questions across key support journeys: KYC, withdrawals, bank details, SIPs, portfolios, family/minor accounts, login, and plan changes.
- Expected source article URL(s).
- Expected answer traits or required facts.
- Refusal/abstention cases.
- Prompt-injection and irrelevant-query cases.

Automated checks should cover:

- Retrieval recall@k: expected source appears in candidate results.
- Citation validity: source URLs exist in the active release.
- Grounding: answer content is supported by supplied context.
- Abstention: unsupported questions return the prescribed escalation response.
- Regression: no material drop from a defined baseline.

Start with 30–50 curated cases. Expand from anonymous, sanitized failure patterns only after a review process exists.

**Initial release gates**

- 95%+ retrieval recall@5 for the curated in-domain set.
- 100% citation URL validity.
- 100% pass rate for designated abstention cases.
- No known high-severity unsupported financial claim in manual review.

### 5. Reliability and observability

**Goal:** Diagnose behaviour without retaining sensitive user questions by default.

Add structured events with a request ID and timestamps for:

- App startup and active KB release.
- Resource/model load success or failure.
- Retrieval duration, candidate count, selected source IDs, and confidence decision.
- Provider selected, fallback reason, time-to-first-token, total duration, and sanitized error class.
- Final response state: success, abstained, retrieval failure, or provider failure.

Do not log raw API keys, prompts, full user messages, or streamed completions in standard logs. If future debugging needs content sampling, introduce an explicit retention policy and user notice first.

Add:

- A lightweight `/health`-equivalent service check for KB availability and configured providers.
- Clear user-facing degraded states: KB unavailable, provider temporarily unavailable, or insufficient KB evidence.
- A simple daily review of error count, provider fallback count, and evaluation results.

### 6. Delivery quality

**Goal:** Make deployments repeatable and changes safe.

Add:

- Python version declaration.
- Fully pinned dependency lockfile with hashes where practical.
- Formatter/linter/type-check configuration.
- Unit tests for prompts, retrieval decisions, configuration, provider fallback, and manifest validation.
- Integration tests using a small fixture KB and mocked provider streams.
- CI checks: formatting, lint, type check, tests, dependency vulnerability scan, and evaluation suite.
- Deployment documentation that distinguishes source code, generated KB releases, and secrets.

Decide one artifact strategy explicitly:

1. Package one immutable KB release with the deployment image, or
2. Store versioned KB releases outside the Git repository and fetch the promoted release during deploy.

Do not continue with generated artifacts ignored yet simultaneously tracked in Git.

## Delivery sequence — Task tracker

> **Legend:** ☐ not started · ◐ in progress · ☑ done

---

### Milestone 1 — Foundation

**Goal:** Extract business logic from `app.py` into clean modules, add tooling, keep the UI working.

#### 1.1 Project setup & tooling

- [x] Declare Python version (`>=3.11`) in `pyproject.toml`.
- [x] Generate a fully pinned dependency lockfile (e.g. `uv lock` or `pip-compile` with hashes).
- [x] Add formatter config (Ruff or Black).
- [x] Add linter config (Ruff).
- [x] Add type-checker config (mypy or pyright, strict on new modules).
- [x] Add `pytest` and `pytest-cov` to dev dependencies.
- [x] Create initial `tests/` directory with `conftest.py`.
- [x] Add a `Makefile` or task runner with `lint`, `format`, `typecheck`, `test` targets.

#### 1.2 Configuration module — `config.py`

Extract from `app.py` (key resolution, constants, model names, health flags):

- [x] Define a typed config dataclass/model (`AppConfig`) for all settings:
  - API keys: `GROQ_API_KEY`, `GEMINI_API_KEY`.
  - Model names: `GROQ_MODEL` (default `openai/gpt-oss-120b`), `GEMINI_MODEL` (default `gemini-2.5-flash`), `EMBED_MODEL` (default `all-MiniLM-L6-v2`).
  - Paths: `CHROMA_DIR`, `ARTICLES_FILE`, `COLLECTION_NAME`.
  - Limits: `TOP_K`, `MAX_INPUT_LENGTH`, `MAX_CONTEXT_SIZE`, `MAX_OUTPUT_TOKENS`.
- [x] Move `_resolve_key()` logic into config module (resolve from `st.secrets` → env var → None).
- [x] Add startup validation: fail fast with clear error if required config is missing or invalid.
- [x] Write unit tests for config loading, defaults, and validation errors.

#### 1.3 Knowledge-base store — `kb_store.py`

Extract from `app.py` `load_resources()` and `retrieve_contexts()`:

- [x] Create `KBStore` class wrapping ChromaDB `PersistentClient` + `SentenceTransformer`.
- [x] Move `load_resources()` logic: ChromaDB init, collection fetch, embed model load, status reporting.
- [x] Move `load_kb_stats()` logic: article count and category list from `articles.json`.
- [x] Move `retrieve_contexts()`: query embedding, `collection.query()`, distance→score mapping.
- [x] Add health-check method: DB exists, collection is non-empty, embed model loaded.
- [x] Classify load errors as typed exceptions (`KBUnavailableError`, `CollectionError`, `ModelLoadError`).
- [x] Write unit tests with a small fixture collection (mocked ChromaDB).

#### 1.4 Prompt construction — `prompting.py`

Extract from `app.py` (system prompt, prompt template, starter questions):

- [x] Move `SYSTEM_PROMPT` constant.
- [x] Move `build_prompt()`: context assembly, 1800-char truncation, template stitching.
- [x] Move starter questions and assistant greeting text.
- [x] Write unit tests for prompt construction (expected format, truncation boundary).

#### 1.5 LLM providers — `providers.py`

Extract from `app.py` (`_groq_stream`, `_gemini_stream`, error handling):

- [x] Define typed provider exceptions (`ProviderRateLimitError`, `ProviderAuthError`, `ProviderUnavailableError`, `ProviderModelNotFoundError`).
- [x] Move `_groq_stream()`: Groq chat completion streaming call.
- [x] Move `_gemini_stream()`: Gemini `generate_content` streaming call.
- [x] Replace string-matching error classification (`"429" in err_text`) with typed exception parsing.
- [x] Extract fallback policy logic: Groq failure → Gemini fallback, with typed reason tracking.
- [x] Write unit tests for fallback logic and error classification (mocked API responses).

#### 1.6 Service orchestration — `service.py`

Extract from `app.py` (query handling, orchestration, validation):

- [x] Create `handle_query()` orchestrator: validate → retrieve → confidence check → prompt → generate → fallback.
- [x] Move input validation (empty query, max input length).
- [x] Add `insufficient_context` decision stub (threshold-based, to be tuned in M2).
- [x] Define response model: `ServiceResponse(answer_stream, sources, provider_used, fallback_reason, status)`.
- [x] Write unit tests for orchestration (mocked kb_store + providers).

#### 1.7 Slim down `app.py`

- [x] Rewire `app.py` to import and call `config`, `kb_store`, `prompting`, `providers`, `service`.
- [x] Remove all extracted business logic from `app.py` (target: ~300 lines of pure Streamlit UI).
- [x] Ensure UI never exposes raw exception text — use sanitized error messages from service layer.
- [x] Manual smoke test: starter questions work, provider fallback works, KB stats sidebar works.

#### 1.8 CI pipeline

- [x] Create GitHub Actions (or equivalent) workflow: format check → lint → type check → tests.
- [x] Ensure CI runs on push and PR.

**Exit condition:** Clean CI run. Local deployment with no UI regression. All starter questions produce correct answers.

---

### Milestone 2 — Trustworthy retrieval

**Goal:** Chunk-level retrieval with confidence thresholds, abstention, citations, and an evaluation suite.

#### 2.1 Chunk-level indexing

- [ ] Design chunk schema: `chunk_id`, `article_id`, `title`, `url`, `category`, `folder`, `section_header`, `content`, `content_hash`, `chunk_position`, `chunk_total`.
- [ ] Implement overlapping semantic chunker (configurable chunk size ~500 tokens, overlap ~100 tokens).
- [ ] Update `build_index.py` to produce chunk-level documents with full metadata.
- [ ] Retain article title/URL/category as metadata on every chunk.

#### 2.2 Retrieval improvements

- [ ] Increase candidate retrieval count (e.g. top-20) then deduplicate by article before passing to LLM.
- [ ] Add configurable distance threshold for `insufficient_context` decision.
- [ ] Replace `1 - distance` "Relevance %" display with transparent "Retrieved sources" presentation.
- [ ] Remove `_score_color` / relevance bar — show source cards without misleading percentages.

#### 2.3 Abstention and citations

- [ ] Implement deterministic `insufficient_context` response when no chunk passes threshold.
- [ ] Add support escalation message for abstention cases.
- [ ] Require LLM response to cite source IDs; map IDs to user-visible article links.
- [ ] Update `prompting.py` with citation instructions in system prompt.

#### 2.4 Evaluation suite

- [ ] Create `eval/` directory with evaluation framework.
- [ ] Write 30–50 curated test cases across: KYC, withdrawals, bank details, SIPs, portfolios, family/minor accounts, login, plan changes.
- [ ] Include expected source article URL(s) per case.
- [ ] Include refusal/abstention cases (out-of-domain, prompt injection).
- [ ] Implement automated checks:
  - [ ] Retrieval recall@k (expected source in candidates).
  - [ ] Citation URL validity (URLs exist in active release).
  - [ ] Abstention pass rate (unsupported queries get escalation response).
- [ ] Run baseline measurement and record results.

**Exit condition:** ≥95% retrieval recall@5 on curated set. 100% citation URL validity. 100% abstention pass rate.

---

### Milestone 3 — Releasable knowledge base

**Goal:** Versioned, staged ingestion with atomic promotion and rollback.

#### 3.1 Release directory format

- [ ] Create `data/releases/` structure with timestamped release dirs.
- [ ] Define `manifest.json` schema: release ID, generation time, crawl times, article/chunk/category counts, source URLs, content hashes, embedder model/version, index schema version, validation result, previous release ID.
- [ ] Implement `current.json` pointer file for active release.

#### 3.2 Staged ingestion workflow

- [ ] Implement staged crawl → normalize → deduplicate pipeline (refactor `scraper.py`).
- [ ] Implement diff against current release (new/changed/removed articles).
- [ ] Build chunk-level index in staged directory.
- [ ] Run validation and retrieval regression tests on staged release.
- [ ] Implement atomic promote: update `current.json` only after validation passes.
- [ ] Retain at least one previous release for rollback.

#### 3.3 Rollback and artifact strategy

- [ ] Implement rollback command: repoint `current.json` to previous release without rebuilding.
- [ ] **Decision:** Package index with deployment image, OR fetch versioned release during deploy.
- [ ] Remove generated artifacts from Git tracking (if choosing external storage).
- [ ] Update `kb_store.py` to load from `current.json`-pointed release directory.

#### 3.4 Scheduled refresh

- [ ] Add CLI command or script for unattended ingestion run.
- [ ] Gate automated promotion on passing validation + eval suite.
- [ ] Document manual review process for first N runs.

**Exit condition:** Failed build leaves active release unaffected. Successful release can be rolled back. Manifest is reproducible.

---

### Milestone 4 — Operable deployment

**Goal:** Structured telemetry, health checks, sanitized errors, and reliability bounds.

#### 4.1 Observability module — `observability.py`

- [ ] Implement structured JSON event logger with request ID and timestamps.
- [ ] Log events: app startup, active KB release, resource/model load, retrieval (duration, candidate count, source IDs, confidence decision), provider (selected, fallback reason, TTFT, total duration, error class), final response state.
- [ ] Ensure raw API keys, prompts, user messages, and completions are NOT logged.

#### 4.2 Health and degraded states

- [ ] Add `/health`-equivalent check: KB availability + configured providers.
- [ ] Implement user-facing degraded state messages: KB unavailable, provider temporarily unavailable, insufficient KB evidence.
- [ ] Sanitize all error messages shown to users (no raw exceptions, no secret-derived details).

#### 4.3 Reliability bounds

- [ ] Enforce maximum input length limit.
- [ ] Enforce maximum retrieved-context size limit.
- [ ] Enforce maximum output-token limit.
- [ ] Add in-process request concurrency limit.

#### 4.4 Operational documentation

- [ ] Document daily review procedure: error count, provider fallback count, eval results.
- [ ] Update README with deployment, update, and rollback runbooks.
- [ ] Document secrets management and configuration for operators.

**Exit condition:** Operator can identify active KB release, provider failures, fallback events, and degraded state from structured logs — without examining raw exceptions.

---

## Risks and decisions needed later

- [ ] **Data governance:** Decide whether sanitized query metadata may be retained and for how long.
- [ ] **Artifact hosting:** Choose packaged index versus external versioned storage before Milestone 3. *(Blocking M3.3)*
- [ ] **Evaluation ownership:** Identify the person/team who signs off support-content changes and financial-safety cases.
- [ ] **Provider policy:** Confirm fallback model behaviour is acceptable for customer-facing answers and determine the desired outage message.
- [ ] **Scale change:** Revisit auth/rate limits/spend controls before expanding beyond the current two users or sharing the public link broadly.

## Definition of done for V2

V2 is complete when the application has a tested service layer, a versioned/rollback-capable KB release process, quality gates for retrieval and abstention, privacy-safe operational telemetry, reproducible builds and CI, and clear documentation for running and updating the small-user deployment.
