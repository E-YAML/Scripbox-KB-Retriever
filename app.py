"""
app.py — Scripbox KB Help Assistant (Production)
A polished, production-grade Streamlit RAG interface for the Scripbox Knowledge Base.
Powered by ChromaDB + sentence-transformers + Groq Llama 3.3-70b (streaming).
Fallback: Google Gemini 2.0 Flash when Groq hits rate limits.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb
import streamlit as st
from groq import Groq
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ─── Page config — MUST be the first Streamlit call ──────────────────────────
st.set_page_config(
    page_title="Scripbox Help Assistant",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://help.scripbox.com",
        "About": "Scripbox KB Assistant — RAG powered by ChromaDB + Groq Llama 3.3",
    },
)

# ─── Constants ────────────────────────────────────────────────────────────────
CHROMA_DIR      = "./chroma_db"
COLLECTION_NAME = "scripbox_kb"
EMBED_MODEL     = "all-MiniLM-L6-v2"
TOP_K           = 5
DEFAULT_GROQ_MODEL = "llama-3.3-70b-specdec"
ARTICLES_FILE   = "./articles.json"

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
    "👋 Hello! I'm your Scripbox Help Assistant. "
    "I can answer questions about investing, KYC, withdrawals, account management, "
    "and more — all sourced directly from the official Scripbox Knowledge Base. "
    "What would you like to know?"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.stApp { background: #FFFFFF; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(170deg, #0D1B2A 0%, #12263F 60%, #0A1A28 100%) !important;
    border-right: 1px solid rgba(0, 166, 147, 0.15) !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] li {
    color: #CBD5E1 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] strong {
    color: #F1F5F9 !important;
}
section[data-testid="stSidebar"] .stMarkdown code {
    background: rgba(0,166,147,0.15) !important;
    color: #00C6AE !important;
    border-radius: 4px;
    padding: 0.15em 0.4em;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
}
/* Clear button in sidebar */
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(239, 68, 68, 0.12) !important;
    border: 1px solid rgba(239, 68, 68, 0.35) !important;
    color: #FCA5A5 !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(239, 68, 68, 0.22) !important;
    border-color: rgba(239, 68, 68, 0.6) !important;
    color: #FEE2E2 !important;
    transform: translateY(-1px) !important;
}

/* ── Main content width ── */
.main .block-container {
    max-width: 860px !important;
    margin: 0 auto !important;
    padding: 1.5rem 2rem 5rem 2rem !important;
}

/* ── Header card ── */
.kb-header {
    background: linear-gradient(135deg, #007A6B 0%, #00A693 50%, #00C6AE 100%);
    border-radius: 16px;
    padding: 1.75rem 2.25rem;
    margin-bottom: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.kb-header::before {
    content: '';
    position: absolute;
    top: -60%;
    right: -5%;
    width: 280px;
    height: 280px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
    pointer-events: none;
}
.kb-header::after {
    content: '';
    position: absolute;
    bottom: -40%;
    right: 18%;
    width: 180px;
    height: 180px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
    pointer-events: none;
}
.kb-header h1 {
    font-size: 1.7rem;
    font-weight: 800;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 1;
}
.kb-header p {
    font-size: 0.92rem;
    opacity: 0.88;
    margin: 0;
    position: relative;
    z-index: 1;
}
.kb-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    backdrop-filter: blur(4px);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 0.75rem;
    position: relative;
    z-index: 1;
}

/* ── Welcome / empty state ── */
.welcome-box {
    text-align: center;
    padding: 1.5rem 1rem 1rem 1rem;
}
.welcome-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.welcome-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.4rem;
}
.welcome-sub {
    font-size: 0.88rem;
    color: #6B7280;
    margin-bottom: 1.5rem;
    line-height: 1.5;
}
.chips-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.6rem;
    text-align: left;
}

/* ── Starter chips ── */
.chip-btn > button,
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: #F8FFFE !important;
    border: 1.5px solid #CCFAF4 !important;
    color: #065F52 !important;
    border-radius: 10px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 0.6rem 0.9rem !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 3rem !important;
    transition: all 0.2s ease !important;
    line-height: 1.4 !important;
    width: 100% !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: #ECFDF8 !important;
    border-color: #00A693 !important;
    color: #00796B !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 14px rgba(0, 166, 147, 0.15) !important;
}

/* ── Chat messages ── */
div[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    margin-bottom: 0.6rem !important;
    padding: 0.85rem 1rem !important;
    animation: msgFadeIn 0.25s ease !important;
}
@keyframes msgFadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Sources section ── */
.sources-header {
    font-size: 0.78rem;
    font-weight: 700;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 1rem 0 0.5rem 0;
}
/* Expander styling for sources */
div[data-testid="stExpander"] {
    border: 1px solid #E5F7F4 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    margin-bottom: 0.5rem !important;
    background: #FAFFFE !important;
    transition: box-shadow 0.2s !important;
}
div[data-testid="stExpander"]:hover {
    box-shadow: 0 2px 12px rgba(0, 166, 147, 0.08) !important;
}
div[data-testid="stExpander"] summary {
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    color: #1F2937 !important;
    padding: 0.6rem 0.85rem !important;
    background: transparent !important;
}
div[data-testid="stExpander"] summary:hover {
    background: #F0FDFB !important;
}
div[data-testid="stExpanderDetails"] {
    background: #FFFFFF !important;
    padding: 0.5rem 0.85rem 0.85rem !important;
}

/* ── Progress bar (relevance score) ── */
div[data-testid="stProgressBar"] > div {
    background: rgba(0,166,147,0.12) !important;
    border-radius: 6px !important;
    height: 6px !important;
}
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #00A693, #00C6AE) !important;
    border-radius: 6px !important;
    height: 6px !important;
}

/* ── Chat input ── */
div[data-testid="stChatInputContainer"] {
    border-top: 1px solid #F3F4F6 !important;
    padding-top: 0.75rem !important;
}
div[data-testid="stChatInputContainer"] > div {
    border: 1.5px solid #D1D5DB !important;
    border-radius: 12px !important;
    background: #FFFFFF !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stChatInputContainer"] > div:focus-within {
    border-color: #00A693 !important;
    box-shadow: 0 0 0 3px rgba(0,166,147,0.12) !important;
}

/* ── Error / warning / info blocks ── */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.88rem !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu  { visibility: hidden; }
footer     { visibility: hidden; }
.stDeployButton { display: none !important; }

/* ── Sidebar stat chips ── */
.stat-row {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}
.stat-pill {
    flex: 1;
    background: rgba(0,166,147,0.12);
    border: 1px solid rgba(0,166,147,0.22);
    border-radius: 10px;
    padding: 0.6rem 0.5rem;
    text-align: center;
}
.stat-num {
    display: block;
    font-size: 1.3rem;
    font-weight: 800;
    color: #00C6AE;
    line-height: 1.1;
}
.stat-lbl {
    display: block;
    font-size: 0.65rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-top: 0.15rem;
}
.status-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.82rem;
    padding: 0.3rem 0;
    color: #CBD5E1;
}
.dot-green { color: #34D399; font-size: 0.65rem; }
.dot-red   { color: #F87171; font-size: 0.65rem; }
.dot-amber { color: #FBBF24; font-size: 0.65rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ─── API Key resolution ────────────────────────────────────────────────────────
def _resolve_key(secret_name: str) -> str:
    """
    Resolve an API key from st.secrets (Streamlit Cloud / local secrets.toml)
    with fallback to environment variable of the same name.
    """
    try:
        key = st.secrets.get(secret_name, "")
        if key:
            return key
    except Exception:
        pass
    return os.getenv(secret_name, "")


GROQ_API_KEY   = _resolve_key("GROQ_API_KEY")
GEMINI_API_KEY = _resolve_key("GEMINI_API_KEY")
GROQ_MODEL     = _resolve_key("GROQ_MODEL") or DEFAULT_GROQ_MODEL


# ─── Cached resources ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    """Load ChromaDB collection and embedding model (cached across sessions)."""
    db_path = Path(CHROMA_DIR)
    if not db_path.exists():
        return None, None, "chroma_missing"

    try:
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        return None, None, f"collection_error:{exc}"

    try:
        embed_model = SentenceTransformer(EMBED_MODEL)
    except Exception as exc:
        return None, None, f"model_error:{exc}"

    return collection, embed_model, "ok"


@st.cache_data(show_spinner=False)
def load_kb_stats() -> dict:
    """Return article count + category list from articles.json (best-effort)."""
    stats: dict = {"total": 0, "categories": 0, "category_list": []}
    try:
        with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
            articles = json.load(f)
        stats["total"] = len(articles)
        cats = sorted({a.get("category", "").strip() for a in articles if a.get("category")})
        stats["categories"] = len(cats)
        stats["category_list"] = cats
    except Exception:
        pass
    return stats


# ─── Helper functions ─────────────────────────────────────────────────────────
def retrieve_contexts(query: str, collection, embed_model) -> list[dict]:
    """Embed query and retrieve top-K matching articles from ChromaDB."""
    qv = embed_model.encode(query).tolist()
    res = collection.query(
        query_embeddings=[qv],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    if res["ids"] and res["ids"][0]:
        for i in range(len(res["ids"][0])):
            score = float(1.0 - res["distances"][0][i])
            hits.append(
                {
                    "title":    res["metadatas"][0][i].get("title", "Untitled"),
                    "url":      res["metadatas"][0][i].get("url", "#"),
                    "category": res["metadatas"][0][i].get("category", ""),
                    "folder":   res["metadatas"][0][i].get("folder", ""),
                    "document": res["documents"][0][i],
                    "score":    score,
                }
            )
    return hits


def build_prompt(query: str, hits: list[dict]) -> str:
    """Construct the RAG prompt from retrieved articles."""
    parts = []
    for i, hit in enumerate(hits, 1):
        clean_doc = hit["document"].replace("\n", " ")[:1800]
        parts.append(
            f"[Article {i}: {hit['title']}]\n"
            f"Category: {hit['category']} > {hit['folder']}\n"
            f"URL: {hit['url']}\n\n"
            f"{clean_doc}"
        )
    context = "\n\n---\n\n".join(parts)
    return f"KNOWLEDGE BASE ARTICLES:\n{context}\n\nUSER QUESTION: {query}\n\nANSWER:"


def _groq_stream(prompt: str):
    """Generator — yields text chunks from Groq streaming API."""
    client = Groq(api_key=GROQ_API_KEY)
    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _gemini_stream(prompt: str):
    """Generator — yields text chunks from Gemini streaming API (fallback)."""
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        "gemini-2.0-flash",
        system_instruction=SYSTEM_PROMPT,
    )
    stream = model.generate_content(prompt, stream=True)
    for chunk in stream:
        if chunk.text:
            yield chunk.text


def _score_color(score: float) -> str:
    if score >= 0.70:
        return "#10B981"  # green
    if score >= 0.50:
        return "#F59E0B"  # amber
    return "#EF4444"      # red


def render_sources(hits: list[dict]):
    """Render source article expanders with relevance bars."""
    st.markdown('<p class="sources-header">📚 Source Articles</p>', unsafe_allow_html=True)
    for i, hit in enumerate(hits, 1):
        score     = max(0.0, min(1.0, hit["score"]))
        score_pct = int(score * 100)
        color     = _score_color(score)
        label     = f"[{i}]  {hit['title']}"

        with st.expander(label):
            meta_col, score_col = st.columns([3, 1])
            with meta_col:
                parts = [p for p in [hit["category"], hit["folder"]] if p]
                st.caption(f"📁 {'  ›  '.join(parts)}" if parts else "📁 General")
            with score_col:
                st.markdown(
                    f'<span style="font-size:0.8rem;font-weight:700;color:{color};">'
                    f"Relevance {score_pct}%</span>",
                    unsafe_allow_html=True,
                )

            st.progress(score)

            preview = hit["document"].replace("\n", " ").strip()
            preview = (preview[:420] + "…") if len(preview) > 420 else preview
            st.markdown(
                f'<p style="font-size:0.83rem;color:#4B5563;line-height:1.55;'
                f'margin:0.6rem 0 0.75rem 0;">{preview}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<a href="{hit["url"]}" target="_blank" '
                f'style="font-size:0.82rem;color:#00A693;font-weight:600;'
                f'text-decoration:none;">Read full article →</a>',
                unsafe_allow_html=True,
            )


# ─── Load resources ───────────────────────────────────────────────────────────
with st.spinner("Loading knowledge base…"):
    collection, embed_model, load_status = load_resources()

kb_stats = load_kb_stats()

# Determine counts for sidebar — prefer articles.json, fall back to collection
total_articles = kb_stats["total"] or (collection.count() if collection else 0)
total_categories = kb_stats["categories"]

db_ok      = load_status == "ok"
groq_ok    = bool(GROQ_API_KEY)
gemini_ok  = bool(GEMINI_API_KEY)
llm_ok     = groq_ok or gemini_ok   # app is usable if either provider is configured

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:

    # Logo + branding
    st.markdown(
        """
        <div style="padding:0 0 0.25rem 0; margin-bottom:1rem;">
            <div style="font-size:1.5rem; font-weight:800; color:#FFFFFF; letter-spacing:-0.5px;">
                💚 Scripbox KB
            </div>
            <div style="font-size:0.75rem; color:#64748B; letter-spacing:0.3px;">
                Knowledge Base Assistant
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # System status
    st.markdown(
        '<div style="font-size:0.7rem;font-weight:700;color:#475569;text-transform:uppercase;'
        'letter-spacing:0.8px;margin-bottom:0.6rem;">System Status</div>',
        unsafe_allow_html=True,
    )
    db_dot     = '<span class="dot-green">●</span>' if db_ok     else '<span class="dot-red">●</span>'
    groq_dot   = '<span class="dot-green">●</span>' if groq_ok   else '<span class="dot-amber">●</span>'
    gemini_dot = '<span class="dot-green">●</span>' if gemini_ok else '<span class="dot-amber">●</span>'

    groq_label   = f'Ready — {GROQ_MODEL}' if groq_ok   else 'No key (rate-limit fallback)'
    gemini_label = 'Ready — Flash 2.0'       if gemini_ok else 'No key (configure for fallback)'

    st.markdown(
        f'<div class="status-row">{db_dot}     &nbsp;Vector DB&nbsp;&nbsp;'
        f'<span style="color:#475569;">{'Connected' if db_ok else 'Not found'}</span></div>'
        f'<div class="status-row">{groq_dot}   &nbsp;Groq (primary)&nbsp;&nbsp;'
        f'<span style="color:#475569;">{groq_label}</span></div>'
        f'<div class="status-row">{gemini_dot} &nbsp;Gemini (fallback)&nbsp;&nbsp;'
        f'<span style="color:#475569;">{gemini_label}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # KB stats
    st.markdown(
        '<div style="font-size:0.7rem;font-weight:700;color:#475569;text-transform:uppercase;'
        'letter-spacing:0.8px;margin-bottom:0.6rem;">Knowledge Base</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-pill">
                <span class="stat-num">{total_articles}</span>
                <span class="stat-lbl">Articles</span>
            </div>
            <div class="stat-pill">
                <span class="stat-num">{total_categories}</span>
                <span class="stat-lbl">Categories</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if kb_stats["category_list"]:
        with st.expander("📂 Browse categories", expanded=False):
            for cat in kb_stats["category_list"]:
                st.markdown(
                    f'<div style="font-size:0.82rem;padding:0.2rem 0;color:#94A3B8;">• {cat}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # Model info
    st.markdown(
        '<div style="font-size:0.7rem;font-weight:700;color:#475569;text-transform:uppercase;'
        'letter-spacing:0.8px;margin-bottom:0.5rem;">Model</div>',
        unsafe_allow_html=True,
    )
    active_model = GROQ_MODEL if groq_ok else "gemini-2.0-flash"
    st.markdown(
        f'<div style="font-size:0.8rem;color:#94A3B8;">🤖 `{active_model}`</div>'
        f'<div style="font-size:0.8rem;color:#94A3B8;margin-top:0.2rem;">↩ Fallback: `gemini-2.0-flash`</div>'
        f'<div style="font-size:0.8rem;color:#94A3B8;margin-top:0.2rem;">🔍 `{EMBED_MODEL}`</div>'
        f'<div style="font-size:0.8rem;color:#94A3B8;margin-top:0.2rem;">Top-K results: `{TOP_K}`</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # API key notices
    if not llm_ok:
        st.error(
            "**No LLM keys configured.**\n\n"
            "Add at least one key to `.streamlit/secrets.toml`:\n"
            "```\nGROQ_API_KEY   = \"gsk_...\"   # console.groq.com\n"
            "GEMINI_API_KEY = \"AIza...\"  # aistudio.google.com\n```"
        )
    else:
        key_info = []
        if groq_ok:   key_info.append("Groq ✓")
        if gemini_ok: key_info.append("Gemini ✓")
        st.markdown(
            f'<div style="font-size:0.75rem;color:#475569;padding:0.25rem 0;">'
            f'🔑 {" · ".join(key_info)} — shared for all users</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Clear conversation
    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE, "hits": []}
        ]
        st.session_state.pending_query = None
        st.rerun()

    # Footer
    st.markdown(
        '<div style="font-size:0.7rem;color:#334155;text-align:center;'
        'margin-top:2rem;padding:0.5rem;border-top:1px solid rgba(255,255,255,0.06);">'
        'Powered by ChromaDB · sentence-transformers<br>Groq Llama 3.3 · Gemini 2.0 Flash</div>',
        unsafe_allow_html=True,
    )


# ─── Session state init ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": WELCOME_MESSAGE, "hits": []}
    ]
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ─── Main content ─────────────────────────────────────────────────────────────

# Header
_badge_provider = "Groq Llama 3.3-70b" if groq_ok else "Gemini 2.0 Flash"
st.markdown(
    f"""
    <div class="kb-header">
        <div class="kb-badge">⚡ Live · {_badge_provider}</div>
        <h1>💚 Scripbox Help Assistant</h1>
        <p>Ask anything about investing, KYC, withdrawals, account management &amp; more —
        answered from the official Scripbox Knowledge Base.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Hard stop if DB not ready ────────────────────────────────────────────────
if not db_ok:
    if "chroma_missing" in load_status:
        st.error(
            "**Knowledge base not found.**\n\n"
            "The `chroma_db/` directory is missing. Please run:\n"
            "```bash\npython scraper.py      # scrape articles (~3 min)\n"
            "python build_index.py  # build vector index (~30 sec)\n```"
        )
    else:
        st.error(f"**Failed to load knowledge base:** `{load_status}`")
    st.stop()

# ── Render chat history ──────────────────────────────────────────────────────
is_first_message = len(st.session_state.messages) == 1  # only welcome msg

for msg in st.session_state.messages:
    avatar = "💚" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("hits"):
            render_sources(msg["hits"])

# ── Starter chips (shown only before first user query) ───────────────────────
if is_first_message:
    st.markdown(
        """
        <div class="welcome-box">
            <div class="chips-label">✨ Try asking</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)
    cols = [col1, col2, col1, col2, col1, col2]
    for i, question in enumerate(STARTER_QUESTIONS):
        if cols[i].button(question, key=f"chip_{i}", use_container_width=True):
            st.session_state.pending_query = question
            st.rerun()

# ── Chat input ───────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "E.g. How do I update my bank account?",
    disabled=not llm_ok,   # disabled only if BOTH keys are missing
)

# Merge chat input with pending query from chips
user_query: str | None = user_input or st.session_state.pending_query
if st.session_state.pending_query:
    st.session_state.pending_query = None

# ── Process query ─────────────────────────────────────────────────────────────
if user_query:
    # Show user bubble
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query, "hits": []})

    # Assistant response
    with st.chat_message("assistant", avatar="💚"):

        # Step 1: Retrieve
        with st.spinner("🔍 Searching knowledge base…"):
            hits = retrieve_contexts(user_query, collection, embed_model)

        if not hits:
            error_msg = (
                "I couldn't find any relevant articles for your question. "
                "Please try rephrasing, or contact "
                "[Scripbox Support](https://help.scripbox.com) directly."
            )
            st.warning(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg, "hits": []}
            )
            st.stop()

        # Step 2: Stream answer — Groq primary, Gemini automatic fallback
        prompt = build_prompt(user_query, hits)
        answer = None

        # ── Try Groq first ────────────────────────────────────────────────────
        if groq_ok:
            try:
                answer = st.write_stream(_groq_stream(prompt))
            except Exception as exc:
                err_text = str(exc)
                is_rate_limit = "429" in err_text or "rate_limit" in err_text.lower()
                is_auth_err   = "401" in err_text or "invalid_api_key" in err_text.lower()
                is_model_err  = "404" in err_text or "model_not_found" in err_text.lower() or "decommissioned" in err_text.lower()

                if (is_rate_limit or is_model_err) and gemini_ok:
                    notice = (
                        "⚡ Groq rate limit reached — seamlessly switching to Gemini fallback…"
                        if is_rate_limit
                        else f"⚠️ Groq model `{GROQ_MODEL}` unavailable or decommissioned — seamlessly switching to Gemini fallback…"
                    )
                    st.info(notice, icon="🔄")
                    # answer will be filled by Gemini block below
                elif is_auth_err:
                    error_msg = (
                        "**Groq authentication error.** The API key is invalid or expired. "
                        "Please update `GROQ_API_KEY` in your Streamlit secrets."
                    )
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg, "hits": []}
                    )
                    st.stop()
                elif is_model_err and not gemini_ok:
                    error_msg = (
                        f"**Groq Model Error:** The model `{GROQ_MODEL}` is not found or has been decommissioned on Groq.\n\n"
                        "Please update `GROQ_MODEL` in your `.env` or `.streamlit/secrets.toml` to an active Groq model "
                        "(e.g., `llama-3.3-70b-specdec` or `llama-3.1-8b-instant`), or configure `GEMINI_API_KEY` for automatic fallback."
                    )
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg, "hits": []}
                    )
                    st.stop()
                elif is_rate_limit and not gemini_ok:
                    error_msg = (
                        "**Groq rate limit reached** and no Gemini fallback is configured.\n\n"
                        "Add `GEMINI_API_KEY` to your Streamlit secrets for automatic fallback. "
                        "Get a free key at [aistudio.google.com](https://aistudio.google.com/apikey)."
                    )
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg, "hits": []}
                    )
                    st.stop()
                else:
                    error_msg = f"**Groq error:** {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg, "hits": []}
                    )
                    st.stop()

        # ── Gemini: used as fallback (or primary if Groq key absent) ──────────
        if answer is None and gemini_ok:
            try:
                answer = st.write_stream(_gemini_stream(prompt))
            except Exception as exc:
                err_text = str(exc)
                if "429" in err_text or "quota" in err_text.lower():
                    error_msg = (
                        "**Both Groq and Gemini have hit their rate limits.**\n\n"
                        "Please wait a few minutes and try again."
                    )
                elif "401" in err_text or "api_key" in err_text.lower():
                    error_msg = (
                        "**Gemini authentication error.** The API key is invalid. "
                        "Please update `GEMINI_API_KEY` in your Streamlit secrets."
                    )
                else:
                    error_msg = f"**Gemini error:** {exc}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "hits": []}
                )
                st.stop()

        if answer is None:
            # Shouldn't reach here, but safeguard
            st.error("No LLM provider available. Please check your API keys.")
            st.stop()

        # Step 3: Render sources inline (below the streamed answer)
        render_sources(hits)

    # Persist to session state
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "hits": hits}
    )
