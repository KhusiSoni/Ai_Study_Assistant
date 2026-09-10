# AI Study Assistant — Context-Aware Doubt Solver

Upload your notes or PDFs, ask questions, and get answers grounded in your own material — built on a **Retrieval-Augmented Generation (RAG)** architecture.

Created by **Khushi**

---

## RAG Architecture

<p align="center">
  <img src="screenshots/architecture.svg" alt="RAG Architecture Diagram" width="800">
</p>

## What This Shows

- **RAG Pipeline** — Semantic document chunking → TF-IDF vector indexing → hybrid retrieval → context-grounded answer generation
- **Product Thinking** — Clean UX flow: upload → index → ask → answer with source attribution
- **Optional AI Enhancement** — Local mode returns top excerpts; `OPENAI_API_KEY` enables GPT-4o-mini answers with inline citations
- **Suite Integration** — Third product in the AI tools suite, matching the dark theme and code conventions of the Resume Analyzer and Code Review Assistant

## Features

| Feature | Detail |
|---------|--------|
| **Semantic Chunking** | Paragraph-aware splitting (500–800 char windows) instead of naive sliding window |
| **Hybrid Retrieval** | TF-IDF cosine similarity (70%) + keyword overlap boosting (30%) |
| **Inline Citations** | OpenAI-powered answers cite source chunks as `[1]`, `[2]` |
| **Streaming** | Real-time token generation via `/api/ask/stream` |
| **Multi-Document** | Upload and manage multiple files per session |
| **Conversation Memory** | Last 6 Q&A pairs passed as context for follow-ups |
| **DOCX Support** | Parse Word documents alongside PDF and TXT |
| **Session Isolation** | Each browser session has its own isolated RAG engine |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.12+, Flask |
| Retrieval | scikit-learn (TF-IDF with bigrams, sublinear tf, l2 norm) |
| Parsing | PyMuPDF, python-docx |
| AI (optional) | OpenAI Chat Completions API (GPT-4o-mini) |
| Frontend | Vanilla JS, CSS3 (dark theme) |

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

### Optional: AI-Powered Answers

Set `OPENAI_API_KEY` to enable GPT-4o-mini answers with inline citations:

```bash
export OPENAI_API_KEY=sk-...
# or on Windows:
# set OPENAI_API_KEY=sk-...
```

Without the API key, the app returns the most relevant excerpts from your documents.

## Usage

1. **Upload** — Drop or select PDF, DOCX, or TXT files in the sidebar
2. **Ask** — Type any question about your study material in the chat input
3. **Review** — Read the answer with source badges; expand "retrieved context" to see which passages were used
4. **Manage** — Remove individual documents or clear all with the sidebar buttons

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/session` | Session info + loaded documents |
| `POST` | `/api/upload` | Upload a file |
| `GET` | `/api/documents` | List all documents |
| `DELETE` | `/api/documents/<id>` | Remove a specific document |
| `POST` | `/api/ask` | Ask a question (JSON response) |
| `POST` | `/api/ask/stream` | Ask a question (NDJSON streaming) |
| `POST` | `/api/clear` | Clear all documents and session |

## Project Structure

```
├── app.py               # Flask routes, API endpoints, streaming
├── rag_engine.py        # RAG pipeline (chunking, TF-IDF, hybrid retrieval)
├── requirements.txt
├── LICENSE
├── static/
│   └── style.css        # Dark theme UI
├── templates/
│   └── index.html       # Document manager + chat interface
├── screenshots/
│   └── architecture.svg  # RAG architecture diagram
└── uploads/             # Uploaded files (gitignored)
```

## License

MIT License — see [LICENSE](LICENSE). Created by **Khushi**.