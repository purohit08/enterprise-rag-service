# DualLens Agent — Design & Reference

A deep dive into [`duallens/agent.py`](agent.py): what it is, how it works, every
design decision, and how to run, extend, and reason about it.

---

## 1. What it is

**DualLens Agent** ranks companies for investment by fusing two lenses:

- **Quantitative** — financial performance (valuation, momentum, risk) from Yahoo Finance.
- **Qualitative** — AI-strategy signal retrieved from each company's AI-initiative report (RAG).

It analyses five companies — **Amazon, Alphabet, IBM, Microsoft, NVIDIA** — and
produces a grounded, cited Buy/Hold/Sell ranking with a single best pick.

### Agent vs. pipeline

The repo ships the same idea in two forms:

| | `DualLens_Analytics.ipynb` | `agent.py` (this doc) |
|---|---|---|
| Control flow | **Fixed pipeline** — cells run top to bottom in a set order | **Agentic** — the model decides what to call and when |
| Who drives | The notebook author | Claude Opus 4.8, via a tool-use loop |
| Lenses | Hard-wired function calls | Exposed as **tools** the model may call, skip, or repeat |
| Transparency | Linear cells | Every tool call is printed as it happens |

The pipeline always does *retrieve → rank*. The agent might retrieve once, or
five times with refined queries, or pull extra metrics for a borderline
company — because it chooses its own path to the answer.

---

## 2. Architecture at a glance

```
          ┌─────────────────────────────────────────────┐
          │              run_agent(question)             │
          │   manual while-loop over the Messages API    │
          └───────────────┬─────────────────────────────┘
                          │  messages[]  (full history each turn)
                          ▼
                 ┌──────────────────┐
                 │  Claude Opus 4.8 │  system prompt = DualLens analyst rules
                 │  (adaptive think)│  tools = [the three below]
                 └───────┬──────────┘
          stop_reason == "tool_use"?         stop_reason == "end_turn"
                 │ yes                               │
                 ▼                                   ▼
   ┌─────────────────────────────┐          final text answer
   │ dispatch each tool_use block│
   └──────┬───────────┬──────────┘
          ▼           ▼            ▼
  get_stock_   get_price_    search_ai_strategy
   metrics     performance    (RAG retrieval)
     │              │                │
   yfinance      yfinance     OpenAI embeddings → Chroma
          │           │            │
          └───────────┴────────────┘
                      ▼
          tool_result blocks appended to messages[]  ──┐
                      │                                 │
                      └──────────── loop back ──────────┘
```

### Why Claude for reasoning + OpenAI for embeddings

Claude has **no embeddings API**, so the stack splits the work:

- **Claude Opus 4.8** (`claude-opus-4-8`) — all reasoning: tool selection, synthesis, ranking.
- **OpenAI `text-embedding-3-small`** — embeddings only, for retrieval.

Both are reachable through gateways (see [§7](#7-configuration)).

---

## 3. The three tools

Each tool is a plain Python function plus a JSON-schema declaration in `TOOLS`.
The model sees only the declaration; `_DISPATCH` maps a tool name to its function.

### `get_stock_metrics(companies?)` — quantitative lens
Fundamental metrics from `yfinance.Ticker(t).info`:

| Field | Yahoo key | Meaning |
|---|---|---|
| `market_cap` | `marketCap` | Company size |
| `pe_ratio` | `trailingPE` | Valuation (price vs. earnings) |
| `dividend_yield` | `dividendYield` | Income |
| `beta` | `beta` | Volatility vs. market |
| `total_revenue` | `totalRevenue` | Scale of business |

Omitting `companies` returns all five.

### `get_price_performance(companies?, period="3y")` — quantitative lens
Momentum. Pulls the close-price history and returns, per company:
`start_price`, `end_price`, and **`total_return_pct`** (normalised, so a $900
stock and a $40 stock are directly comparable).

### `search_ai_strategy(query, top_k=4)` — qualitative lens
RAG retrieval over the AI-initiative PDFs. Embeds the query with OpenAI, runs a
cosine search over Chroma, and returns the top passages tagged with the source
company. This is how the agent grounds any claim about a company's AI strategy —
the system prompt forbids asserting such claims from memory.

---

## 4. The agent loop

[`run_agent()`](agent.py) is a manual, framework-free loop — no LangChain, no
agent executor. In pseudocode:

```python
messages = [user question]
for turn in range(MAX_TURNS):            # safety cap = 12
    resp = claude.messages.create(
        model="claude-opus-4-8",
        thinking={"type": "adaptive"},   # Opus 4.8: the only on-mode
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
    )
    print any text blocks                # live transparency

    if resp.stop_reason != "tool_use":
        return final text                # done

    messages.append(assistant turn)      # includes thinking + tool_use blocks
    for each tool_use block:
        run _DISPATCH[name](**input)     # errors come back as is_error results
    messages.append(all tool_result blocks in ONE user message)
```

Key correctness points:

- **Full assistant content is echoed back** (`resp.content`), not just text — so
  thinking blocks and `tool_use` blocks are preserved for the next turn, as the
  API requires.
- **Parallel tool calls** are supported: one assistant turn can emit several
  `tool_use` blocks; all their `tool_result`s go back in a **single** user
  message (splitting them would train the model out of parallel calls).
- **Tool errors don't crash the loop** — they're returned to the model as
  `tool_result` with `is_error: true`, so it can recover or route around them.
- **`MAX_TURNS` cap** prevents a runaway loop; hitting it returns a clear message.

---

## 5. The system prompt (the agent's rules)

`SYSTEM_PROMPT` defines DualLens's behaviour. In essence:

1. **Fuse two lenses** — quantitative financials and qualitative AI strategy.
2. **Ground every AI-strategy claim** in a `search_ai_strategy` result — never from memory.
3. **Cite both a number and a specific AI initiative** for each company discussed.
4. For a ranking: end with a **Buy/Hold/Sell** per company and one clearly labelled **single best pick**, one line of reasoning each.
5. Be concise and concrete; name the source company for each passage.

Rules 2–3 are what make the output trustworthy: numbers are always from the
quant tools, strategy claims are always from retrieved text.

---

## 6. Retrieval internals

| Step | Detail |
|---|---|
| **Load** | `pypdf.PdfReader` extracts text from each of the 5 PDFs |
| **Chunk** | Sliding window — `CHUNK_SIZE=800`, `CHUNK_OVERLAP=150` chars, whitespace-normalised |
| **Embed** | OpenAI `text-embedding-3-small`, in batches of 128 chunks |
| **Store** | Chroma `PersistentClient` at `duallens/chroma_agent/`, cosine space |
| **Metadata** | Each chunk carries `company`, `ticker`, `chunk` index |
| **Retrieve** | Query embedded the same way; `TOP_K=4` cosine nearest neighbours |

`ingest()` is **idempotent** — chunk IDs are `"{ticker}::{i}"`, so re-ingesting
upserts (overwrites) rather than duplicating. A first run with an empty store
auto-ingests before answering.

---

## 7. Configuration

Keys load from a `.env` at the **repo root** (gitignored). All optional overrides
are read from the environment.

| Variable | Required | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Claude Opus 4.8 reasoning |
| `OPENAI_API_KEY` | ✅ | Embeddings (OpenAI or a compatible gateway) |
| `OPENAI_BASE_URL` | optional | Point embeddings at a proxy/gateway instead of `api.openai.com` |
| `ANTHROPIC_BASE_URL` | optional | Point Claude at a gateway instead of `api.anthropic.com` |
| `ANTHROPIC_WORKSPACE_ID` | optional | Sent as the `anthropic-workspace-id` header when the key isn't workspace-scoped |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | optional | Enable Langfuse tracing (see §7b) |

Tunable constants live at the top of `agent.py`: `LLM_MODEL`, `EMBED_MODEL`,
`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `MAX_TOKENS`, `MAX_TURNS`, `COMPANIES`.

### 7b. Observability (optional Langfuse tracing)

If `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` are set and `langfuse` is
installed, each run is traced: a root span `duallens-agent`, a **generation** per
Claude turn (`claude-turn-N`, with token usage), and a **span** per tool call
(`tool:<name>`). Traces flush at the end of the run. If the keys are unset it's a
**complete no-op** — `_langfuse()` returns `None`, `_span()` yields nothing, and
the agent runs identically. The observability layer is fully decoupled from the
core loop. See the full stack in [STACK.md](STACK.md).

---

## 8. Running it

```bash
# one-time environment
uv venv --python 3.12
uv pip install -r duallens/requirements.txt

# keys
cp .env.example .env      # then fill in ANTHROPIC_API_KEY + OPENAI_API_KEY

# build the vector store (once, or after changing the PDFs)
python -m duallens.agent --ingest

# run the agent
python -m duallens.agent                                   # default: rank all 5
python -m duallens.agent "Is NVIDIA a buy given its AI strategy?"
python -m duallens.agent --quiet "Compare IBM and Microsoft for income"
```

**CLI flags:** `--ingest` (build store and exit) · `--quiet` (print only the
final answer, suppress the live tool trace).

---

## 9. Design decisions & trade-offs

- **Framework-free loop.** The loop is readable in one screen and has no hidden
  control flow — consistent with the repo's "explicit and inspectable" ethos.
  The cost is that niceties (retries, tracing) are hand-rolled if needed.
- **Opus 4.8 specifics.** Adaptive thinking is the only on-mode; `temperature`
  and `budget_tokens` are rejected (400), and assistant prefill is unsupported —
  so none are used. Output format is steered by the system prompt, not prefill.
- **Local Chroma + per-tool embedding.** The vector store is on-disk and the
  agent embeds queries itself, so retrieval is self-contained and reproducible.
- **Gateway-friendly.** Base-URL and workspace-ID env hooks let the same code run
  against OpenAI/Anthropic directly or through a classroom/enterprise gateway.
- **Safety cap over open-ended looping.** `MAX_TURNS=12` bounds cost and
  guarantees termination.

---

## 10. Limitations & future scope

- **Live financials.** `yfinance` is unofficial and rate-limited; some fields can
  be missing. For production, swap in a licensed data provider behind the same tool.
- **No grounding score yet.** The notebook's LLM-as-Judge (groundedness +
  relevance, 1–5) isn't wired into the agent. A `--eval` flag could score each
  retrieval-backed answer and gate low-confidence claims.
- **Retrieval precision.** Dense-only search; hybrid (dense + keyword) retrieval
  and a re-ranker would raise precision on narrow queries.
- **Fixed universe.** Five companies, five PDFs. Broadening the universe means
  adding PDFs and tickers to `COMPANIES`, then re-ingesting.
- **Not financial advice.** Output is an analytical demonstration, not a
  personalised recommendation.
