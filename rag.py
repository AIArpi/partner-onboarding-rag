# =============================
# File: rag.py
# Purpose: Retrieval + Generation helpers.
#          1) retrieve(): nearest‑neighbor search over your vector store
#          2) answer():   prompt a small local LLM (Ollama) using retrieved context
# Run:     imported by app.py (you can also test in a Python REPL)
# =============================

# The IBM Granite 2B and 8B models are 128K context length language models that have been 
# fine-tuned for improved reasoning and instruction-following capabilities.
#
# Intended Use
# These models are designed to handle general instruction-following tasks and can be 
# integrated into AI assistants across various domains, including business applications.
#
# Capabilities
# - Thinking
# - Summarization
# - Text classification
# - Text extraction
# - Question-answering
# - Retrieval Augmented Generation (RAG)
# - Code related tasks
# - Function-calling tasks
# - Multilingual dialog use cases
# - Fill-in-the-middle
# - Long-context tasks including long document/meeting summarization, long document QA, etc.
#
# Supported Languages
# English, German, Spanish, French, Japanese, Portuguese, Arabic, Czech, Italian, Korean, 
# Dutch, and Chinese.


import os
import requests
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from typing import List, Dict, Tuple

# System instruction keeps responses concise and grounded in supplied context.
SYSTEM_PROMPT = (
    "You are a precise Partner Program assistant."
    " Answer ONLY using the provided context excerpts."
    " If the answer is not in the context, say: 'I don't know based on the current documentation.'"
    " Be concise and use bullet points. Include a short 'Sources' list with file names."
)


def _get_collection():
    """Open the same Chroma collection created by ingest.py."""
    client = chromadb.PersistentClient(path=os.path.join("data", "db"))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection("partner_docs", embedding_function=embedder)


def retrieve(query: str, k: int = 4) -> List[Tuple[str, Dict]]:
    """Return top‑k (document_text, metadata) pairs for a user query."""
    coll = _get_collection()
    res = coll.query(query_texts=[query], n_results=max(1, k))
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return list(zip(docs, metas))


def _ollama_generate(prompt: str, model: str = None, host: str = None, timeout: int = 120) -> str:
    """Call a local Ollama model (default from .env). Returns plain text.
    Make sure Ollama is running and a model is pulled, e.g.: `ollama pull phi3`.
    """
    load_dotenv()
    model = model or os.getenv("OLLAMA_MODEL", "granite3.3:8b")
    host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
    url = f"{host}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


def answer(query: str, top_k: int = 4) -> str:
    """Compose a grounded answer using retrieved context and a strict system instruction.
    Returns Markdown with a Sources section.
    """
    ctx_pairs = retrieve(query, k=top_k)
    if not ctx_pairs:
        return ("I couldn't retrieve any context. Please run `python ingest.py` after placing your documents "
                "under `data/docs/`, then try again.")

    # Build a compact context block the model can digest.
    # Each chunk is labeled with its source filename for human‑readable citations.
    ctx_text = "\n\n---\n".join(
        [f"[From {m.get('source','unknown')}]:\n{d}" for d, m in ctx_pairs]
    )

    user_prompt = (
        f"Context excerpts:\n{ctx_text}\n\n"
        f"Question: {query}\n\n"
        f"Answer in bullets. Then add a 'Sources' list with the file names only."
    )

    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\nAnswer:"  # final cue
    try:
        return _ollama_generate(full_prompt)
    except Exception as e:
        return ("LLM call failed. Check that Ollama is running and `OLLAMA_HOST`/`OLLAMA_MODEL` are set.\n\n"
                f"Error: {e}")