import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import urllib.request
import json
import os

# Set up page configurations
st.set_page_config(
    page_title="Scripbox Knowledge Base",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="expanded"
)


# Constants
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "scripbox_kb"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant for Scripbox, an investment platform. "
    "Answer the user's question using ONLY the information from the provided knowledge base articles. "
    "If the articles don't contain enough information, say so honestly. "
    "Always cite which article(s) you used at the end of your answer."
)

# --- CACHED RESOURCES ---
@st.cache_resource(show_spinner="Loading Knowledge Base & Models... please wait.")
def load_resources():
    if not os.path.exists(CHROMA_DIR):
        st.error(f"Error: {CHROMA_DIR} not found. Ensure the vector database is present.")
        st.stop()
    
    # Load Chroma
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        st.error(f"Collection error: {e}")
        st.stop()
        
    # Load Embedding Model
    embed_model = SentenceTransformer(EMBED_MODEL)
    return collection, embed_model

collection, embed_model = load_resources()

# --- HELPER FUNCTIONS ---
def retrieve_contexts(query: str):
    qv = embed_model.encode(query).tolist()
    res = collection.query(
        query_embeddings=[qv],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )
    hits = []
    # Make sure we got results
    if res["ids"] and len(res["ids"][0]) > 0:
        for i in range(len(res["ids"][0])):
            hits.append({
                "title": res["metadatas"][0][i].get("title", "No Title"),
                "url": res["metadatas"][0][i].get("url", "#"),
                "category": res["metadatas"][0][i].get("category", ""),
                "folder": res["metadatas"][0][i].get("folder", ""),
                "document": res["documents"][0][i],
                "score": 1.0 - res["distances"][0][i]
            })
    return hits

def build_prompt(query: str, hits: list) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        clean_doc = hit["document"].replace("\n", " ")[:1500]
        parts.append(f"[Article {i}: {hit['title']}]\nURL: {hit['url']}\n{clean_doc}")
    context = "\n\n---\n\n".join(parts)
    return f"KNOWLEDGE BASE ARTICLES:\n{context}\n\nUSER QUESTION: {query}\n\nANSWER:"

def generate_answer(provider: str, api_key: str, prompt: str) -> str:
    if provider == "Groq":
        from groq import Groq
        try:
            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"**Groq Error:** {e}"

    elif provider == "OpenAI":
        from openai import OpenAI
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"**OpenAI Error:** {e}"
            
    elif provider == "Gemini":
        import google.generativeai as genai
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            resp = model.generate_content(f"{SYSTEM_PROMPT}\n\n{prompt}")
            return resp.text
        except Exception as e:
            return f"**Gemini Error:** {e}"

    elif provider == "Ollama (Local)":
        payload = json.dumps({
            "model": "llama3",
            "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read())
                return data.get("response", "Empty Ollama response")
        except Exception as e:
            return f"**Ollama Error:** Is Ollama running? {e}"

    return "Unknown provider."

# --- UI LAYOUT ---

with st.sidebar:
    st.title("⚙️ LLM Settings")
    
    st.markdown("Choose the AI that answers your questions.")
    provider_option = st.selectbox(
        "LLM Provider",
        ("Groq", "OpenAI", "Gemini", "Ollama (Local)")
    )
    
    api_key_input = ""
    if provider_option != "Ollama (Local)":
        api_key_input = st.text_input(f"{provider_option} API Key", type="password")
        if not api_key_input:
            st.warning(f"Please provide your {provider_option} API key.")
        
        st.markdown(f"*{provider_option} runs in the cloud. We don't store your API key!*")
    else:
        st.info("Ollama runs locally on your machine. Ensure it is running (`ollama run llama3`).")

st.title("💸 Scripbox Help Assistant")
st.markdown("Ask anything about Scripbox investing, KYC, withdrawals, and more! Your answer will be sourced straight from the official Scripbox Knowledge Base.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with Scripbox today?"}]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_query := st.chat_input("E.g., How do I update my bank account?"):
    # Add user message to UI
    with st.chat_message("user"):
        st.markdown(user_query)
        
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Assistant responding
    with st.chat_message("assistant"):
        if provider_option != "Ollama (Local)" and not api_key_input:
            st.error(f"Missing {provider_option} API Key in the sidebar!")
            st.stop()
            
        with st.spinner("Searching knowledge base..."):
            hits = retrieve_contexts(user_query)
            
        if not hits:
            st.warning("No relevant articles found in the Scripbox KB.")
            st.stop()
            
        with st.spinner(f"Synthesizing answer using {provider_option}..."):
            prompt = build_prompt(user_query, hits)
            answer = generate_answer(provider_option, api_key_input, prompt)
            
        # Compile response with citations
        full_response = answer + "\n\n**---\nSource Articles:**\n"
        for i, hit in enumerate(hits, 1):
            full_response += f"{i}. [{hit['title']}]({hit['url']}) · *Relevance: {int(hit['score']*100)}%*\n"
            
        st.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
