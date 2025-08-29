# =============================
# File: ingest.py
# Purpose: Read your partner docs (TXT/PDF), chunk them, and store
#          embeddings in a local ChromaDB for retrieval-augmented generation (RAG).
# Run:    python ingest.py
# Notes:  Put your files under data/docs/ (can be nested). Re-run after adding files.
# =============================

import os
import glob
from typing import List, Tuple
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

# ---------- Chunking parameters ----------
# Chunking splits long documents into overlapping slices so the retriever can
# fetch the most relevant passages for a question. You can tweak sizes later.
CHUNK_SIZE = 800       # characters per chunk (≈ 150–200 words)
CHUNK_OVERLAP = 120    # overlap between consecutive chunks (keeps context continuity)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Greedy fixed-size chunking with overlap.
    Simple, reliable, and fast for MVPs. You can upgrade to sentence-aware
    chunking later if needed.
    """
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        chunk = text[i : i + size]
        if chunk.strip():
            chunks.append(chunk)
        i += max(1, size - overlap)
    return chunks


def read_txt(path: str) -> str:
    """Read a UTF‑8 (or best‑effort) text file and normalize whitespace."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()
    # Normalize pesky whitespace so retrieval and display are cleaner
    return "\n".join(line.strip() for line in txt.replace("\r", "").split("\n"))


def read_pdf(path: str) -> str:
    """Extract text from a PDF via PyPDF. Not all PDFs contain extractable text.
    For image‑only PDFs, consider OCR (e.g., Tesseract) as a future enhancement.
    """
    reader = PdfReader(path)
    out_lines = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        out_lines.append(txt)
    return "\n".join(out_lines)


def collect_files(root: str) -> List[str]:
    """Recursively list files from root. Supports .txt and .pdf by default."""
    patterns = ["**/*.txt", "**/*.pdf"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(root, p), recursive=True))
    return [f for f in files if os.path.isfile(f)]


def ensure_sample_copy():
    """Make sure the sample FAQ ships into data/docs/ on first run."""
    src = os.path.join("samples", "sample_partner_faq.txt")
    dst_dir = os.path.join("data", "docs")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    if os.path.exists(src) and not os.path.exists(dst):
        import shutil
        shutil.copy(src, dst)


def build_collection() -> chromadb.api.models.Collection.Collection:
    """Open (or create) the persistent Chroma collection with a sentence‑transformer embedder."""
    # Persistent DB path for Chroma (lives on disk under data/db)
    client = chromadb.PersistentClient(path=os.path.join("data", "db"))
    # A small, fast, free embedding model — great default for local work
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    # Using a stable collection name lets the app find it consistently
    collection = client.get_or_create_collection(
        name="partner_docs", embedding_function=embedder
    )
    return collection


def index_files(files: List[str]) -> Tuple[int, int]:
    """Read, chunk, and upsert files into Chroma. Returns (#files, #chunks)."""
    collection = build_collection()

    # Optional: clear existing entries to avoid duplicates on re‑ingest.
    # Comment this out if you prefer incremental adds.
    try:
        collection.delete(where={})
    except Exception:
        pass

    texts, metadatas, ids = [], [], []

    for fpath in files:
        ext = os.path.splitext(fpath)[1].lower()
        try:
            raw = read_pdf(fpath) if ext == ".pdf" else read_txt(fpath)
        except Exception as e:
            print(f"[skip] {fpath}: {e}")
            continue
        for j, ch in enumerate(chunk_text(raw)):
            texts.append(ch)
            metadatas.append({"source": os.path.basename(fpath), "path": fpath, "chunk": j})
            ids.append(f"{os.path.basename(fpath)}-{j}")

    if texts:
        collection.add(documents=texts, metadatas=metadatas, ids=ids)
    return len(files), len(texts)


if __name__ == "__main__":
    # Environment is optional here, but load it for consistency across scripts.
    load_dotenv()

    ensure_sample_copy()
    docs_dir = os.path.join("data", "docs")
    os.makedirs(docs_dir, exist_ok=True)

    files = collect_files(docs_dir)
    n_files, n_chunks = index_files(files)

    if n_files == 0:
        print("No documents found. Drop TXT/PDF files under data/docs/ and re‑run: python ingest.py")
    else:
        print(f"Indexed {n_chunks} chunks from {n_files} files into Chroma (collection: partner_docs).")
