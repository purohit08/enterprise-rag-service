# Company AI-Initiatives — RAG Agent

Local Retrieval-Augmented Generation over 5 company PDFs (AMZN, GOOGL, IBM,
MSFT, NVDA). Everything runs on-device — no API keys, no external calls.

Built **framework-free** on purpose — the full RAG pipeline (parse → chunk →
embed → store → retrieve → generate) is written directly against Chroma and
Hugging Face so every step is explicit and inspectable, rather than hidden
behind an orchestration library.

**Stack:** FastAPI + Uvicorn (backend) · Streamlit (UI) · Chroma (vector DB) ·
Hugging Face embeddings + LLM · PyPDF · PyTorch (Apple MPS).

## What this demonstrates
- End-to-end RAG without an orchestration framework — retrieval and prompting
  are hand-wired, not abstracted away.
- A clean service boundary: FastAPI backend (`/ingest`, `/query`) decoupled
  from the Streamlit UI.
- Fully local, reproducible inference on Apple Silicon (MPS).

## Roadmap
This is **v1**. Next it becomes an *Enterprise RAG Service*: a LangChain
ingestion pipeline, additional data sources, hybrid retrieval + re-ranking, a
multi-model router (Azure OpenAI / OpenAI / Claude), **Langfuse** tracing, and a
**promptfoo** eval suite reporting accuracy, hallucination rate, and conciseness.

```
app/
  config.py          # models, paths, chunk sizes
  rag.py             # PDF -> chunk -> embed -> Chroma -> retrieve -> generate
  main.py            # FastAPI endpoints
ui/
  streamlit_app.py   # web UI
Companies-AI-Initiatives/   # the 5 source PDFs
```

## Setup (one time)

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Run

**1. Backend** (terminal 1):
```bash
uv run uvicorn app.main:app --reload
```

**2. Index the PDFs** (once, or after adding PDFs):
```bash
curl -X POST http://localhost:8000/ingest
```

**3. UI** (terminal 2):
```bash
uv run streamlit run ui/streamlit_app.py
```

Then open the Streamlit URL, click **Ingest / Re-index** if you haven't, and
ask questions.

## Models

| Role | Model | Notes |
|------|-------|-------|
| Embeddings | `BAAI/bge-small-en-v1.5` | small & fast |
| Generation | `Qwen/Qwen2.5-3B-Instruct` | ~6 GB, downloads on first query |

Change them in `app/config.py`.
