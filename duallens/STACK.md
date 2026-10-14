# DualLens Agent — Stack & Components

Every external API, model, datastore, and library the agent
([`agent.py`](agent.py)) depends on, and exactly what each is used for.

---

## At a glance

| Layer | Component | Used for |
|---|---|---|
| **Reasoning LLM** | Claude **Opus 4.8** (`claude-opus-4-8`) | All agent reasoning: tool selection, synthesis, ranking |
| **Embedding model** | OpenAI **`text-embedding-3-small`** (1536-dim) | Embedding PDF chunks + queries for retrieval |
| **Vector DB** | **Chroma** (`PersistentClient`, local disk) | Storing embeddings + metadata, cosine k-NN search |
| **Financial data API** | **Yahoo Finance** via `yfinance` | Quantitative lens: metrics + price history |
| **LLM API** | **Anthropic Messages API** (`anthropic` SDK) | Tool-use loop transport to Claude |
| **Embeddings API** | **OpenAI API** (`openai` SDK, OpenAI-compatible) | Embedding calls (direct or via gateway) |
| **Observability** | **Langfuse** (optional) | Tracing runs, Claude calls, tool calls, token usage |
| **PDF parsing** | `pypdf` | Extracting text from the 5 source PDFs |

---

## APIs the agent calls

### 1. Anthropic Messages API — reasoning
- **SDK:** `anthropic` (Python)
- **Endpoint:** `POST /v1/messages` (via `client.messages.create`)
- **Model:** `claude-opus-4-8`
- **Role:** Runs the agentic tool-use loop — Claude decides which tools to call,
  reads results, and produces the final ranking.
- **Request specifics:** `thinking={"type": "adaptive"}` (Opus 4.8's only on-mode),
  `max_tokens=4096`, `tools=[…]`. **No `temperature`, no prefill** — both rejected by Opus 4.8.
- **Auth/overrides:** `ANTHROPIC_API_KEY`; optional `ANTHROPIC_BASE_URL` (gateway)
  and `ANTHROPIC_WORKSPACE_ID` (header for unscoped keys).

### 2. OpenAI Embeddings API — retrieval
- **SDK:** `openai` (Python), OpenAI-compatible
- **Endpoint:** `POST /v1/embeddings` (via `client.embeddings.create`)
- **Model:** `text-embedding-3-small` (1536 dimensions)
- **Role:** Turns PDF chunks (at ingest) and the agent's search queries (at query
  time) into vectors. **Why OpenAI and not Claude? Claude has no embeddings API.**
- **Auth/overrides:** `OPENAI_API_KEY`; optional `OPENAI_BASE_URL` to route through
  an OpenAI-compatible gateway (e.g. a Great Learning endpoint) instead of `api.openai.com`.

### 3. Yahoo Finance — quantitative data
- **Library:** `yfinance` (unofficial Yahoo Finance client; no key required)
- **Role:** Backs two tools:
  - `get_stock_metrics` → `Ticker(t).info` (market cap, trailing P/E, dividend
    yield, beta, total revenue).
  - `get_price_performance` → `Ticker(t).history(period)` (start/end price, total return %).
- **Note:** Unofficial and rate-limited; some fields can be missing. A production
  build would swap in a licensed data provider behind the same tool interface.

### 4. Langfuse — observability (optional)
- **SDK:** `langfuse` (Python)
- **Role:** When `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` are set, traces each
  run: a root span, a **generation** per Claude call (model, token usage, output),
  and a **span** per tool call. Unset → complete no-op; the agent runs unchanged.
- **Auth:** `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
  (e.g. `https://cloud.langfuse.com` or a self-hosted URL).

---

## Models

| Model | Type | Dimensions / size | Where |
|---|---|---|---|
| `claude-opus-4-8` | Instruct LLM (reasoning) | — | Anthropic API |
| `text-embedding-3-small` | Embedding model | 1536-dim | OpenAI API |

> **Local-path alternatives** (used by the sibling `app/` service, not the agent):
> embedding `BAAI/bge-small-en-v1.5` (384-dim) and LLM `Qwen/Qwen2.5-3B-Instruct`,
> both running on-device via Hugging Face — a fully offline, no-key path.

---

## Vector database — Chroma

| Property | Value |
|---|---|
| Engine | Chroma (`chromadb.PersistentClient`) |
| Location | `duallens/chroma_agent/` (on local disk, gitignored) |
| Collection | `duallens_agent` |
| Distance | Cosine (`hnsw:space: "cosine"`) |
| Index | HNSW (approximate nearest-neighbour) |
| Record | `id` + document text + embedding + metadata |
| Metadata | `company`, `ticker`, `chunk` index |
| ID scheme | `"{ticker}::{i}"` → re-ingest **upserts** (idempotent) |
| Retrieval | `top_k = 4` nearest neighbours per query |

**Ingestion params:** chunk size `800` chars, overlap `150`, embedded in batches of 128.

---

## Key Python libraries

| Library | Purpose |
|---|---|
| `anthropic` | Claude Messages API client |
| `openai` | Embeddings client (OpenAI-compatible) |
| `chromadb` | Local vector store |
| `yfinance` | Yahoo Finance data |
| `pypdf` | PDF text extraction |
| `python-dotenv` | Load `.env` keys |
| `langfuse` | Optional tracing/observability |
| `ragas` | RAG-metric evaluation suite (see [EVALUATION.md](EVALUATION.md)) |
| `deepeval` | Eval framework — RAG metrics + GEval rubrics |
| `pandas` | (pipeline) tabular financials |
| `matplotlib` | (pipeline) charts |
| `langchain*` | the fixed-pipeline version (`duallens/pipeline/`) |

---

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Claude reasoning |
| `OPENAI_API_KEY` | ✅ | Embeddings |
| `OPENAI_BASE_URL` | optional | Embeddings gateway (else `api.openai.com`) |
| `ANTHROPIC_BASE_URL` | optional | Claude gateway (else `api.anthropic.com`) |
| `ANTHROPIC_WORKSPACE_ID` | optional | `anthropic-workspace-id` header for unscoped keys |
| `LANGFUSE_PUBLIC_KEY` | optional | Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | optional | Langfuse tracing |
| `LANGFUSE_HOST` | optional | Langfuse endpoint (cloud or self-hosted) |

---

## Data flow (who talks to what)

```
PDFs ──pypdf──> chunks ──OpenAI embeddings API──> vectors ──> Chroma (local)
                                                                  │
user question ──> Claude (Anthropic API) ──decides──> tools ──────┤
                     ▲                                  │         │
                     │                      yfinance ◄──┤         │
                     │                  (Yahoo Finance)  │         │
                     │           OpenAI embeddings API ◄─┴─ query ─┘
                     │                  (retrieval)      │
                     └───────── tool results ────────────┘
                     (every step optionally traced to Langfuse)
```
