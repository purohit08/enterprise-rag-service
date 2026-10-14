# Interview Prep — RAG, Agents & Evals (this repo)

A study guide for talking about this project in an interview. Each concept has:
**what it is → how *this repo* does it → what to say → likely follow-ups.**

The repo has two things to point to:
- **`app/`** — a framework-free local RAG service (FastAPI + Chroma + Hugging Face).
- **`duallens/`** — a dual-lens investment agent (Claude Opus 4.8 tool-use loop + RAG).

> One-line pitch: *"I built a local RAG service over company PDFs, then turned it
> into a tool-using agent that fuses quantitative financial data with qualitative
> AI-strategy signal retrieved from documents, and graded its answers with an
> LLM-as-judge."*

---

## 1. RAG (Retrieval-Augmented Generation)

**What it is.** Instead of relying on what an LLM memorised, you *retrieve*
relevant text from your own data and put it in the prompt as context, then the
model *generates* an answer grounded in that context. It reduces hallucination,
lets you use private/fresh data, and gives you citations.

**How this repo does it.** Pipeline is: **parse PDF → chunk → embed → store in a
vector DB → retrieve top-k by similarity → stuff into the prompt → generate.**
`app/rag.py` does this end-to-end against Chroma; `duallens/agent.py` exposes the
retrieve step as a tool (`search_ai_strategy`) the agent calls on demand.

**What to say.** *"RAG has two phases: an offline ingestion phase (chunk + embed +
index) and an online phase (embed the query, nearest-neighbour search, generate
with the retrieved context). The model only answers from retrieved text, and I
enforce that in the system prompt."*

**Follow-ups:**
- *Why not just fine-tune?* RAG is cheaper, updates instantly when docs change,
  and gives traceable citations; fine-tuning bakes knowledge in and is costly to refresh.
- *How do you stop hallucination?* Instruction to use only the context + say "I
  don't know", citations, and an eval (LLM-as-judge) for groundedness.

---

## 2. Embeddings

**What it is.** A model that turns text into a fixed-length vector of numbers so
that *semantically similar* text lands close together in vector space. That's
what makes "search by meaning" (not keywords) possible.

**How this repo does it.**
- Local service (`app/`): `BAAI/bge-small-en-v1.5` via `sentence-transformers`, runs on-device.
- Agent (`duallens/`): OpenAI `text-embedding-3-small` (1536-dim) — **because
  Claude has no embeddings API**, so OpenAI handles embeddings and Claude does reasoning.
- Vectors are **L2-normalised** so cosine similarity = dot product.

**What to say.** *"Embeddings map text to a vector; similar meanings are nearby.
I normalise the vectors and use cosine distance. The same model must embed both
documents and queries — you can't mix embedding models between index and query."*

**Follow-ups:**
- *Dimensionality?* 384 for bge-small, 1536 for text-embedding-3-small. Higher
  isn't always better — trade-off of recall vs. speed/storage.
- *Cosine vs. Euclidean?* Cosine ignores magnitude, focuses on direction/meaning —
  standard for text embeddings.

---

## 3. Chunking

**What it is.** Splitting documents into smaller passages before embedding, so
retrieval returns focused context instead of a whole 20-page PDF.

**How this repo does it.** Sliding-window character chunks: **800 chars with 150
overlap**. Overlap stops an answer from being cut in half at a chunk boundary.

**What to say.** *"Chunk size is a trade-off: too big dilutes relevance and wastes
context tokens; too small loses surrounding meaning. I use ~800 chars with 150
overlap so facts that straddle a boundary still appear whole in one chunk."*

**Follow-ups:**
- *Token vs. character chunking?* Characters are simplest; token-based or
  sentence/recursive splitting respects structure better (the pipeline uses
  LangChain's `RecursiveCharacterTextSplitter`).
- *How pick the size?* Empirically, via the eval — measure retrieval quality.

---

## 4. Vector database (Chroma)

**What it is.** A store built for vectors: it indexes embeddings and answers
"give me the k nearest vectors to this query vector" fast, using an approximate
nearest-neighbour index (HNSW).

**How this repo does it.** **Chroma** `PersistentClient` on local disk, cosine
space (`hnsw:space: "cosine"`). Each record = id + document text + embedding +
metadata (`company`, `ticker`, `chunk`). Ingestion is **idempotent**: ids are
`"{ticker}::{i}"`, so re-running **upserts** instead of duplicating.

**What to say.** *"The vector DB stores the embedding plus the original text and
metadata. At query time it does ANN search via HNSW and returns the top-k with
distances. I keep metadata so I can filter by company and cite the source."*

**Follow-ups:**
- *HNSW?* Hierarchical Navigable Small World graph — approximate NN, fast at scale.
- *Chroma vs. FAISS/Pinecone/pgvector?* Chroma is simple + local for a demo;
  Pinecone/pgvector/Weaviate for managed/production scale.
- *Metadata filtering?* Yes — e.g. restrict search to one company's chunks.

---

## 5. The LLM (generation) & model choice

**What it is.** The model that reads the retrieved context + question and writes
the answer.

**How this repo does it.**
- Local service: `Qwen/Qwen2.5-3B-Instruct` via Hugging Face `transformers`, on
  Apple Silicon (MPS), fully offline, greedy decoding.
- Agent: **Claude Opus 4.8** (`claude-opus-4-8`) via the Anthropic Messages API.

**Opus 4.8 specifics worth knowing.** Adaptive thinking is the only "on" mode;
`temperature` and `budget_tokens` are rejected (HTTP 400); assistant **prefill is
unsupported** — so output format is steered by the system prompt, not prefill.

**What to say.** *"I kept a fully local path with a small open model for an
offline, no-key demo, and a cloud path with Claude Opus 4.8 for the agent's
harder reasoning. Choosing the model is about capability vs. cost vs. latency vs.
data-residency."*

---

## 6. FastAPI service layer

**What it is.** A Python web framework for building typed HTTP APIs quickly;
Pydantic validates request/response bodies; Uvicorn is the ASGI server.

**How this repo does it.** `app/main.py` exposes:
`GET /health`, `POST /ingest`, `POST /upload` (PDF upload), `POST /query`
(RAG answer). The RAG logic lives in `app/rag.py` — a **clean service boundary**:
the API layer is thin, the UI (Streamlit) just calls the endpoints.

**What to say.** *"I separated the API from the RAG core so the pipeline is
testable and reusable, and the UI is decoupled. Models load lazily on first use
so the server boots instantly."*

**Follow-ups:**
- *Why lazy-load models?* A 3B model is GBs — lazy loading keeps startup fast and
  the process light until first query.
- *Sync vs async?* FastAPI supports async; heavy model calls are CPU/GPU-bound so
  they'd go in a worker/queue for real concurrency.

---

## 7. Agents & tool use

**What it is.** An *agent* is an LLM that, in a loop, decides which **tools**
(functions) to call to accomplish a goal, reads the results, and keeps going
until it can answer. The difference from a pipeline: *the model chooses the
path*, you don't hard-code it.

**How this repo does it.** `duallens/agent.py` is a **manual tool-use loop** over
the Messages API (no framework). Three tools: `get_stock_metrics`,
`get_price_performance` (quant, Yahoo Finance), and `search_ai_strategy` (qual,
RAG). The loop:

1. Send messages + tool definitions to Claude.
2. If `stop_reason == "tool_use"`, run the requested tool(s), append the
   `tool_result`(s), loop.
3. If not, return the final answer.

**What to say.** *"I gave the model tools for each lens and let it decide what to
call. It often calls tools in parallel and does follow-up retrievals on its own.
The loop is a plain `while` so every decision is explicit — I return tool errors
back to the model as `is_error` results so it can recover, and I cap turns to
guarantee termination."*

**Key technical points (say these — they signal depth):**
- **Echo the full assistant content back**, including thinking + `tool_use`
  blocks, or the next turn breaks.
- **Parallel tool calls** → return *all* `tool_result`s in **one** user message.
- **`tool_choice`/loop termination** via `stop_reason`; safety cap `MAX_TURNS`.

**Follow-ups:**
- *Agent vs. workflow?* Use an agent only when the task is open-ended and worth
  the extra cost/latency; otherwise a fixed workflow is cheaper and more predictable.
- *How define a tool?* Name + description + JSON input schema; the description is
  the "prompt" that teaches the model when to use it.

---

## 7b. Tool calling (function calling) — the mechanics

**What it is.** The protocol by which an LLM asks your code to run a function.
You declare tools (name + description + JSON input schema); the model, when it
decides a tool is needed, returns a structured `tool_use` request with arguments
matching the schema; you execute it and feed the result back as a `tool_result`.
The model never runs code itself — it *requests* a call and you *fulfil* it.

**The round-trip (as in this repo):**

1. **Declare** — `TOOLS` is a list of `{name, description, input_schema}`. The
   **description is effectively a prompt** — it teaches the model *when* to call
   the tool, so it's worth writing carefully.
2. **Model decides** — response comes back with `stop_reason == "tool_use"` and
   one or more `tool_use` blocks, each with an `id`, a `name`, and validated
   `input` (JSON).
3. **You execute** — `_DISPATCH[name](**input)` runs the Python function.
4. **You return** — a `tool_result` block referencing the same `tool_use_id`,
   with the output serialised (JSON here). On failure, set `is_error: true`.
5. **Loop** — the model reads results and either calls more tools or answers.

**Things interviewers probe:**
- **Schema / validation.** The `input_schema` is JSON Schema. `strict: true`
  guarantees arguments validate exactly. Always `json.loads` / parse inputs —
  never string-match the serialised JSON (escaping varies).
- **Parallel calls.** One turn can emit several `tool_use` blocks; return **all**
  `tool_result`s in **one** user message, or you train the model to stop
  parallelising.
- **Forcing a call.** `tool_choice` can be `auto` (model decides), `any` (must use
  some tool), or a named tool — though newest models restrict forcing and prefer
  `auto` + a prompt instruction.
- **Error handling.** Return tool failures to the model (`is_error`) instead of
  crashing — it can retry or route around.
- **Termination.** Loop on `stop_reason`; cap turns (`MAX_TURNS`) to bound cost
  and guarantee it ends.

**What to say.** *"Tool calling is a structured request/response: the model emits
a schema-validated call, I execute it, return a tool_result keyed by id, and loop
on stop_reason. The tool's description is really a prompt that teaches the model
when to use it. I handle parallel calls, return errors back as tool results, and
cap the loop."*

---

## 7c. MCP (Model Context Protocol)

**What it is.** MCP is an **open standard (from Anthropic) for connecting LLMs to
tools and data through a uniform interface** — think "USB-C for AI tools." Instead
of hand-wiring every integration into each app, you run an **MCP server** that
exposes capabilities, and any **MCP client** (Claude Desktop, an IDE, your agent)
can discover and use them over a standard protocol (JSON-RPC, over stdio or HTTP).

**The three primitives an MCP server exposes:**
| Primitive | What it is | Analogy |
|---|---|---|
| **Tools** | Functions the model can call (with side effects) | POST endpoints |
| **Resources** | Readable data/context (files, rows, docs) | GET endpoints |
| **Prompts** | Reusable prompt templates the user can invoke | Slash commands |

**Why it matters (the pitch).** Before MCP, every tool integration was bespoke
and locked to one app (the "M×N problem": M apps × N tools = M×N integrations).
MCP turns it into **M+N** — write a server once, every MCP-capable client can use
it; build a client once, it can use every server. It standardises **discovery**
(clients ask a server what it offers), **invocation**, and **auth**.

**MCP vs. plain tool calling (know this distinction).**
- **Tool calling** (what `agent.py` does) is the *model-level* mechanism: the LLM
  requests a function, your code runs it. The tools are defined **in your app**.
- **MCP** is a *transport/standard* that sits around tools: it lets tools live in a
  **separate server/process** and be reused across many apps, with discovery and
  auth built in. Under the hood an MCP tool still surfaces to the model as a normal
  tool call — MCP standardises *where tools come from and how they're wired up*,
  not how the model requests them.

**How it relates to this repo.** `agent.py` defines its three tools **in-process**
— the simplest form. The natural MCP evolution: wrap `search_ai_strategy`,
`get_stock_metrics`, and `get_price_performance` as an **MCP server**, so the same
tools are reusable from Claude Desktop, an IDE, or any other agent without copying
code. (This coding session itself used MCP servers — e.g. Gmail, Drive, Apify —
exposed to the assistant over MCP, which is a concrete example to mention.)

**What to say.** *"MCP is an open standard for plugging tools and data into LLMs
over a uniform protocol — servers expose tools, resources, and prompts, and any
MCP client can discover and call them. It solves the M×N integration problem: you
build a tool server once and reuse it everywhere. My agent defines tools
in-process today; the MCP version would move them into a standalone server so
they're reusable across clients, with standardised discovery and auth."*

**Follow-ups:**
- *Transport?* JSON-RPC 2.0 over stdio (local) or HTTP/SSE (remote).
- *Tools vs resources?* Tools *do* things (side effects, like calling an API);
  resources *provide* read-only context (like a file or DB rows).
- *Security concerns?* A server can run arbitrary code and read data — so trust,
  sandboxing, auth, and least-privilege scopes matter; treat untrusted server
  output as data, not instructions (prompt-injection risk).
- *MCP vs. an agent framework?* MCP is the *connectivity layer* (how tools are
  exposed); the agent loop/framework is the *reasoning layer* (how the model uses
  them). They compose.

---

## 8. Evaluation & LLM-as-judge

**What it is.** Measuring answer quality automatically. **LLM-as-judge** uses a
model to score another model's output against criteria — cheaper and faster than
human grading, good for regression testing.

**How this repo does it.** Two harnesses (full detail in [EVALUATION.md](EVALUATION.md)):
1. **Custom LLM-judge** (`eval.py`, `pipeline/judge.py`) — scores answers 1–5 on
   **groundedness** and **relevance**, returning strict JSON.
2. **RAGAS suite** (`ragas_eval.py`) — the recognised RAG-metrics library:
   - **faithfulness** — hallucination (claims unsupported by context) *[no label]*
   - **answer_relevancy** — does it address the question? *[no label]*
   - **context_precision** — were retrieved passages relevant? *[no label]*
   - **context_recall** — did retrieval cover the reference? *[needs gold]*
   - **conciseness** — a rubric (`AspectCritic`, 0/1): full but not padded
   - **correctness_rubric** — a graded 1–5 rubric (`RubricsScore`) vs the reference

**Rubrics** are how you score *qualities* (conciseness, correctness) that aren't a
simple support check — you hand the judge an explicit scoring rubric. `AspectCritic`
gives a binary verdict against a definition; `RubricsScore` gives a graded 1–5 score
against per-score descriptions.

**What to say.** *"I evaluate on two levels: a hand-rolled LLM-judge for groundedness
and relevance, and the RAGAS suite for the standard RAG metrics — faithfulness,
answer relevancy, context precision and recall — plus rubric-based conciseness and a
1–5 correctness rubric against a gold set. The no-label metrics catch hallucination
and retrieval noise; the gold-set metrics catch misses and measure accuracy. I use
Claude as the judge with a clear rubric to limit bias."*

**Follow-ups:**
- *Isn't the judge biased?* Yes — mitigate with a stronger/separate judge model,
  a clear rubric, few-shot anchors, and spot-checking against human labels.
- *Other RAG metrics?* Context precision/recall (retrieval quality), faithfulness
  (answer vs. context), answer relevance. Tools: RAGAS, promptfoo, DeepEval.
- *Offline vs. online eval?* Offline on a fixed test set before shipping; online
  via user feedback / A-B once live.

---

## 8b. Observability & tracing (Langfuse)

**What it is.** LLM apps are non-deterministic and multi-step, so you need to
*see* what happened: every model call, every tool call, the prompts, the outputs,
latency, token usage, and cost — per request, over time. **Observability** is
that visibility; **tracing** is the structured record of a single run as a tree
of spans. **Langfuse** is an open-source LLM-observability platform for exactly this.

**Core concepts.**
- **Trace** — one end-to-end run (one agent invocation).
- **Span** — a step within it (e.g. a tool call), nested to show structure.
- **Generation** — a special span for an LLM call, capturing model, input, output,
  and **token usage** (→ cost).

**How this repo does it.** `agent.py` has **optional** Langfuse tracing: if
`LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` are set, each run records a root span
(`duallens-agent`), a **generation** per Claude turn (`claude-turn-N`, with token
usage), and a **span** per tool call (`tool:search_ai_strategy`, …). If the keys
aren't set it's a **complete no-op** — the agent runs unchanged. Traces are
flushed at the end of the run.

**What to say.** *"I instrumented the agent with Langfuse so every run is a trace:
a generation per Claude call with token usage, and a span per tool call. That
gives me latency, cost, and the full prompt/response for debugging and for
spotting regressions. I made it optional and zero-overhead when it's off, so it
never couples the core logic to the observability layer."*

**Follow-ups:**
- *Why not just log?* Plain logs lose the *structure* (which tool, inside which
  turn, inside which run) and the aggregates (cost/latency trends). Tracing gives
  a tree and a dashboard.
- *Observability vs. evals?* Complementary: **evals** measure *quality* on a test
  set (groundedness, relevance); **observability** shows *what actually happened*
  in production (latency, cost, errors, the real prompts). Langfuse also stores
  traces you can later turn into eval datasets.
- *OpenTelemetry?* Langfuse v3+ is built on OTEL spans, so it interops with standard
  tracing tooling.
- *Alternatives?* LangSmith, Phoenix/Arize, Helicone, OpenLLMetry — same idea.

---

## 9. The "dual-lens" idea (the project's story)

**What it is.** Fusing **quantitative** signal (numbers: P/E, beta, 3-yr return)
with **qualitative** signal (strategy text from reports) to rank companies — a
screener that sees both the financials *and* how future-ready a company is.

**How this repo does it.** The agent pulls metrics + momentum from Yahoo Finance,
retrieves AI-strategy passages via RAG, and Claude fuses both into a Buy/Hold/Sell
ranking where each call must cite **both a number and a specific AI initiative**.

**What to say.** *"Traditional screens look only at financials and miss
innovation capacity, which lives in unstructured strategy documents. RAG makes
that qualitative signal queryable, and the agent fuses it with the quantitative
lens into one grounded, cited ranking."*

---

## 10. Rapid-fire Q&A cheat sheet

- **What is RAG in one line?** Retrieve relevant context from your data, then
  generate an answer grounded in it (with citations).
- **Why embeddings + vector DB?** Semantic search — find text by meaning, not keywords.
- **What's chunking and why overlap?** Split docs for focused retrieval; overlap
  keeps boundary-straddling facts intact.
- **Cosine vs Euclidean?** Cosine (direction/meaning) is standard for text.
- **What makes it an *agent*?** The LLM chooses which tools to call in a loop,
  rather than following a fixed script.
- **How do you define a tool?** name + description + JSON schema; description tells
  the model *when* to use it.
- **How do you handle tool errors?** Return them to the model as `is_error`
  results; it recovers or routes around.
- **What is tool/function calling?** The model emits a schema-validated `tool_use`
  request; you execute it and return a `tool_result` keyed by id; loop on `stop_reason`.
- **What is MCP?** An open standard to connect LLMs to tools/data over a uniform
  protocol — servers expose **tools, resources, prompts**; any client can use them.
- **MCP vs tool calling?** Tool calling is how the *model* requests a function;
  MCP is the *standard* for where tools live and how they're wired up/discovered
  — solves the M×N integration problem (build once, reuse everywhere).
- **How do you evaluate?** LLM-as-judge on groundedness + relevance (structured
  JSON), aiming for a gold test set and RAG metrics.
- **Biggest risk and mitigation?** Hallucination → context-only prompting,
  citations, groundedness eval.
- **Local vs cloud models?** Local small model for offline/no-key demo; Claude
  Opus 4.8 for the agent's reasoning — trade capability vs cost vs data-residency.
- **How would you productionise?** Managed vector DB, hybrid retrieval + re-ranker,
  tracing (Langfuse), eval suite (promptfoo/RAGAS) in CI, caching, auth, a worker
  queue for model calls.
- **What is observability / tracing?** Structured visibility into each run — a
  **trace** of **spans** (tool calls) and **generations** (LLM calls with token
  usage), for latency/cost/debugging. This repo uses **Langfuse** (optional).
- **Observability vs evals?** Evals measure *quality* on a test set; observability
  shows *what actually happened* at runtime (cost, latency, real prompts).

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation |
| **Embedding** | Text → vector capturing meaning |
| **Vector DB** | Store + ANN index for embeddings (here: Chroma) |
| **HNSW** | Approximate nearest-neighbour graph index |
| **Chunk** | A passage of a document, embedded as one unit |
| **top-k** | The k most similar chunks returned for a query |
| **Cosine similarity** | Similarity by vector angle (ignores magnitude) |
| **Tool / function calling** | LLM invoking a defined function with JSON args |
| **MCP** | Model Context Protocol — open standard to connect LLMs to tools/data |
| **MCP server / client** | Server exposes tools/resources/prompts; client (agent/IDE) consumes them |
| **tool_use / tool_result** | The request block the model emits / the result block you return |
| **Agent loop** | Repeated model→tool→result cycle until done |
| **LLM-as-judge** | Using an LLM to score another model's output |
| **Observability** | Runtime visibility into an LLM app (cost, latency, prompts) |
| **Trace / span / generation** | A run / a step / an LLM-call step within it (Langfuse) |
| **Langfuse** | Open-source LLM observability & tracing platform |
| **Groundedness** | Are the answer's claims supported by the context? |
| **Adaptive thinking** | Opus 4.8 mode where the model decides how much to reason |
| **ASGI / Uvicorn** | Async server standard / server running FastAPI |
