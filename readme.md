# 🤝 Partner Onboarding RAG — Local, Private, Fast

&#x20; &#x20;

> A lightweight agentic AI & **Retrieval‑Augmented Generation (RAG)  for partner‑program enablement.** Ask about MDF, enablement assets, deal registration SLAs, onboarding steps. Answers are grounded **only** in your uploaded PDFs/TXTs.

---

## Why this matters (for channel leaders & execs)

- **Self‑service enablement**: partners get accurate answers 24/7.
- **Single source of truth**: standardizes answers across regions and teams.
- **Private by default**: runs locally via **Ollama** + **ChromaDB**.

**Hiring‑manager view**: demonstrates RAG fundamentals, prompt guardrails, transparent retrieval traces, and pragmatic, deployable UX.

---

## Demo assets

- 📄 **Test run (PDF):** [Partner Onboarding RAG Test Q&A.pdf](https://github.com/AIArpi/partner-onboarding-rag/blob/main/samples/Partner%20Onboarding%20RAG%20Test%20Q%26A.pdf) — a captured Q&A session for quick review.



---

## Features

- **Local, free stack**: Streamlit UI • ChromaDB vector store • Sentence‑Transformer embeddings (`all-MiniLM-L6-v2`) • Small LLM via **Ollama**.
- **Grounded answers**: strict prompt *“answer only from context; otherwise say I don’t know.”*
- **Transparency**: expandable panel shows the retrieved passages (with file names) used to answer.
- **Fast iteration**: drop new PDFs/TXTs → re‑ingest in seconds.

---

## Quickstart

### 0) Prereqs

- Python **3.10+**
- **Ollama** installed and running; pull a small model (default: `phi3`).
  ```bash
  ollama pull granite3.3:8b   # or another local model you've installed (e.g., command-r-7b)
  ```

### 1) Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If scripts are blocked: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force`

### 2) Ingest your documents

```powershell
# Put PDFs/TXTs into data\docs\
python ingest.py
```

### 3) Run the app

```powershell
python -m streamlit run app.py
```

Open the browser link Streamlit prints.

---

## Configuration

Copy `.env.example` → `.env` to override defaults:

```
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=granite3.3:8b  # e.g., command-r-7b if you pulled it
TOP_K=4                     # number of passages to retrieve per question
```

> Tested with`command-r-7b` (fast, small) and `granite3.3:8b` (higher quality on longer instructions). Switch by editing `OLLAMA_MODEL`.

---

## Usage

1. Drop partner docs into `data/docs/` (nested folders fine).
2. Run `python ingest.py` to (re)index.
3. Start: `python -m streamlit run app.py`.
4. Ask questions; expand **Show retrieved context** to see exactly what grounded the answer.

**Tip**: increase **Top‑K** in the sidebar if answers feel incomplete.

---

## Architecture (at a glance)

```
User question → Vector retrieval (Chroma, MiniLM embeddings) →
Context snippets → Prompt w/ guardrails → Local small LLM (Ollama) → Answer + Sources
```

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector store:** ChromaDB (persistent at `data/db/`)
- **LLM:** any Ollama model (`OLLAMA_MODEL`), default `phi3`

---

## Project structure

```
partner-onboarding-rag/
  app.py                 # Streamlit chat UI
  ingest.py              # PDF/TXT → chunks → Chroma (vector DB)
  rag.py                 # retrieval + LLM generation with guardrails
  requirements.txt
  .env.example
  data/
    docs/                # your source PDFs/TXTs (git-ignored)
    db/                  # vector store (git-ignored; .gitkeep kept)
  samples/
    sample_partner_faq.txt
  docs/
    test-run.pdf         # your captured Q&A
    screenshot.png       # optional UI screenshot
```

---

## Reproduce the test PDF (optional)

1. Run the app and ask a short suite of questions (MDF, enablement links, deal reg SLAs).
2. Save the transcript (e.g., copy/paste to a doc and export to PDF, or take screenshots).
3. Place it at samples/Partner Onboarding RAG Test Q&A.pdf and commit:

```powershell
git add samples/Partner Onboarding RAG Test Q&A.pdf
git commit -m "samples: add test run PDF"
git push
```

---

## Extending this project

- **Citations with highlighting**: render in‑text citations that link back to source chunks.
- **File uploader**: ingest new docs from the UI (no restarts).
- **Collections per partner**: `partner_docs_<region>` with a dropdown switcher.
- **OCR pipeline**: handle image‑only PDFs via Tesseract.
- **Deployment**: Dockerfile + `docker compose` to ship internally.

---

## Troubleshooting (Windows)

- **Script execution blocked**: run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force`.
- **Ollama not reachable**: check service and port; update `OLLAMA_HOST` in `.env`.
- **No answers**: ensure files are in `data/docs/` and re‑run `python ingest.py`.
- **Clear & re‑ingest**: delete `data/db/` contents and run `python ingest.py`.

---

## Security & privacy

- All retrieval and generation runs locally by default.
- Only local HTTP calls to Ollama; no docs are uploaded externally.

---

## License

MIT — see [`LICENSE`](LICENSE).

## Author

**Arpad Bihami** — [LinkedIn](https://www.linkedin.com/in/arpadbihami)

> Hiring‑manager note: this repo demonstrates practical RAG with strong guardrails, transparent retrieval traces, and a clean, minimal UI — optimized for partner enablement use‑cases.

