# =============================
# File: app.py
# Purpose: Streamlit chat UI for the Partner Onboarding RAG bot.
# Run:     streamlit run app.py
# =============================

import os
import streamlit as st
from dotenv import load_dotenv
from rag import answer, retrieve

# ---------- Streamlit page config ----------
st.set_page_config(page_title="Partner Onboarding RAG", page_icon="🤝", layout="wide")
st.title("🤝 Partner Onboarding Bot (RAG)")

# ---------- Environment / options ----------
load_dotenv()
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-K passages", 1, 10, DEFAULT_TOP_K)
    st.caption("Increase if answers feel incomplete; decrease for speed.")
    st.markdown(
        "**Docs folder:** `data/docs/`  "+
        "Add TXT/PDF files there → run `python ingest.py` → ask questions here."
    )

st.write(
    "Ask anything about your **partner program**: MDF, enablement assets, deal registration SLAs, onboarding steps, etc."
)

# Previous conversation (optional for nicer UX)
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, message)

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

# ---------- Chat input ----------
q = st.chat_input("Ask about MDF, enablement, deal reg…")
if q:
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking with your docs…"):
            a = answer(q, top_k)
            st.markdown(a)

        # Optional: show retrieved chunks for transparency/debugging
        with st.expander("Show retrieved context excerpts"):
            ctx = retrieve(q, k=top_k)
            for i, (doc, meta) in enumerate(ctx, start=1):
                st.markdown(f"**#{i} — {meta.get('source','unknown')} (chunk {meta.get('chunk')})**")
                st.code(doc)

    # save to history (so it persists on rerun)
    st.session_state.history.append(("user", q))
    st.session_state.history.append(("assistant", a))

st.caption(
    "Powered by local embeddings (all-MiniLM-L6-v2) + your docs in ChromaDB + a small local LLM via Ollama (phi3 by default)."
)
