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
duallens/
  DualLens_Analytics.ipynb  # dual-lens screener (fixed pipeline)
  agent.py                  # dual-lens screener as a tool-using agent
Companies-AI-Initiatives/   # the 5 source PDFs
```

## DualLens Agent — agentic RAG (`duallens/agent.py`)

Where `DualLens_Analytics.ipynb` runs a fixed pipeline, `agent.py` turns the two
lenses into **tools** and lets **Claude Opus 4.8** run a manual tool-use loop,
deciding what to call and when. Framework-free on purpose: the agent loop is a
plain `while` over the Messages API, so every tool call is explicit and printed.

| Tool | Lens | Source |
|------|------|--------|
| `get_stock_metrics` | Quantitative | Yahoo Finance (cap, P/E, yield, beta, revenue) |
| `get_price_performance` | Quantitative | 3-yr normalised return (momentum) |
| `search_ai_strategy` | Qualitative | RAG over the 5 AI-initiative PDFs (OpenAI embeddings → Chroma) |

```bash
# keys: ANTHROPIC_API_KEY (reasoning) + OPENAI_API_KEY (embeddings) in .env at repo root.
# optional: OPENAI_BASE_URL / ANTHROPIC_BASE_URL (gateways), ANTHROPIC_WORKSPACE_ID.
uv pip install -r duallens/requirements.txt
python -m duallens.agent --ingest          # one-time: build the vector store
python -m duallens.agent                   # rank all 5 companies
python -m duallens.agent "Is NVIDIA a buy given its AI strategy?"
```

Optional [Langfuse](https://langfuse.com) tracing: set `LANGFUSE_PUBLIC_KEY` /
`LANGFUSE_SECRET_KEY` and every run is traced (Claude calls + tool calls + token
usage); unset, it's a no-op.

**Docs:** [`duallens/AGENT.md`](duallens/AGENT.md) (design & reference) ·
[`duallens/STACK.md`](duallens/STACK.md) (APIs, models, embedding model, vector DB) ·
[`duallens/INTERVIEW_PREP.md`](duallens/INTERVIEW_PREP.md) (concept study guide).

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
