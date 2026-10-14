# DualLens — Evaluation

How the agent is evaluated, which metrics are used, and how to run them. There
are two complementary harnesses:

| Harness | File | Metrics | Dependency |
|---|---|---|---|
| **Custom LLM-as-judge** | [`eval.py`](eval.py) | groundedness, relevance (1-5) | none (raw Anthropic SDK) |
| **RAGAS suite** | [`ragas_eval.py`](ragas_eval.py) | faithfulness, answer relevancy, context precision/recall, conciseness, correctness rubric | `ragas` |
| **DeepEval framework** | [`deepeval_eval.py`](deepeval_eval.py) | faithfulness, answer relevancy, contextual precision/recall, + GEval conciseness & correctness rubrics | `deepeval` |

All use **Claude Opus 4.8** as the evaluator (judge); RAGAS also uses the
project's OpenAI embeddings. Same gateway/workspace config as everything else.

These map onto the common "AI agent eval" landscape: **RAG + response evals**
(RAGAS), an **eval framework** (DeepEval), and **tracing + observability**
(Langfuse, see [AGENT.md](AGENT.md) §7b). Three well-chosen tools for a RAG agent,
not one of every kind.

---

## Why two harnesses?

- **`eval.py`** is framework-free and dependency-light — it shows you understand
  what an LLM-judge *is* by implementing one (capture evidence → judge answer).
- **`ragas_eval.py`** uses **RAGAS**, the recognised RAG-metrics library, giving
  the standard named metrics an interviewer expects, plus rubric-based scores.

---

## The metrics

### RAGAS suite (`ragas_eval.py`)

| Metric | Range | Needs gold label? | What it catches |
|---|---|---|---|
| **faithfulness** | 0-1 | No | **Hallucination** — claims in the answer not supported by retrieved context |
| **answer_relevancy** | 0-1 | No | Off-topic or partial answers (does it address the question?) |
| **context_precision** | 0-1 | No | Retrieval **noise** — were the retrieved passages actually relevant? |
| **context_recall** | 0-1 | **Yes** | Retrieval **misses** — did retrieval cover what the reference answer needs? |
| **conciseness** | 0 or 1 | No | Bloat — a *rubric* (`AspectCritic`): full answer without padding |
| **correctness_rubric** | 1-5 | **Yes** | **Accuracy** — a graded *rubric* (`RubricsScore`) vs the reference answer |

- **No-label metrics** judge the answer against the evidence the agent itself
  retrieved (internal consistency). **Gold-label metrics** judge against a
  reference answer (ground truth) — see the `GOLD` set in `ragas_eval.py`.
- **Rubrics** are how you evaluate *qualities* (conciseness, correctness) that
  aren't a simple support check: you give the judge an explicit scoring rubric.
  - `AspectCritic` → binary 0/1 against a definition (conciseness here).
  - `RubricsScore` → graded 1-5 against per-score descriptions (correctness here).

### DeepEval framework (`deepeval_eval.py`)

A pytest-style eval framework. Same RAG metrics (its own LLM-judge
implementations) plus **GEval** — DeepEval's custom-criteria metric where you
state a rubric in plain English and it scores with chain-of-thought.

| Metric | Range | Needs gold? | Notes |
|---|---|---|---|
| **faithfulness** | 0-1 | No | claims vs retrieval_context |
| **answer_relevancy** | 0-1 | No | addresses the question |
| **contextual_precision** | 0-1 | **Yes** | relevant passages ranked first |
| **contextual_recall** | 0-1 | **Yes** | retrieval covers the expected answer |
| **conciseness** (GEval) | 0-1 | No | rubric: full but not padded |
| **correctness** (GEval) | 0-1 | **Yes** | rubric: matches the expected answer |

Claude is wired in as a **custom `DeepEvalBaseLLM`** (`_build_claude_model`), so
DeepEval judges with Opus 4.8 instead of its default OpenAI judge — the wrapper
returns schema-validated JSON for the metrics that need structured output.

### Custom judge (`eval.py`)

| Metric | Range | What it catches |
|---|---|---|
| **groundedness** | 1-5 | Same idea as faithfulness — claims supported by gathered evidence |
| **relevance** | 1-5 | Does the answer address the question? |

---

## How evidence is captured

Evaluating an *agent* (not a single RAG call) needs the evidence it actually
gathered. Both harnesses pass an `on_event` callback to `agent.run_agent`, which
streams `tool`/`tool_result` events. They collect the passages returned by
`search_ai_strategy` as the retrieved context, and use the agent's final text as
the answer. RAGAS then scores (question, retrieved_contexts, answer, reference).

---

## Running

```bash
# Custom LLM-as-judge
python -m duallens.eval                 # built-in question set
python -m duallens.eval -q "Is IBM a buy?"

# RAGAS suite (uses the gold set in ragas_eval.py)
python -m duallens.ragas_eval           # all gold questions
python -m duallens.ragas_eval -n 2      # first N (cheaper; each run makes many judge calls)

# DeepEval framework (same gold set)
python -m duallens.deepeval_eval        # all gold questions
python -m duallens.deepeval_eval -n 2   # first N
```

RAGAS writes per-question detail to `duallens/ragas_results.csv`; the custom
judge writes `duallens/eval_results.json`. Both are gitignored.

### Example RAGAS scorecard (2 gold questions)

```
faithfulness:        0.91   # low hallucination
answer_relevancy:    0.86
context_precision:   0.96   # retrieved passages were relevant
context_recall:      1.00   # retrieval covered the reference
conciseness:         0.50   # ranking-style answers run long
correctness_rubric:  5.0    # answers matched the reference
```

---

## Notes & honest caveats (good interview material)

- **No-label metrics measure self-consistency, not truth.** faithfulness/
  precision can be high while the answer is still wrong if retrieval was wrong.
  The **gold set** (context_recall, correctness_rubric) is what grounds the
  scores in truth — currently small; expand it for stronger accuracy numbers.
- **Judge bias/leniency.** One model judging itself can inflate scores. Mitigate
  with a clear rubric (done), a separate/stronger judge, and spot-checking
  against human labels.
- **Conciseness is real signal here (0.50).** The agent's Buy/Hold/Sell answers
  are thorough but verbose — a lever to tune in the system prompt.
- **Opus 4.8 + RAGAS gotchas handled in code:** `temperature` is rejected by
  Opus 4.8, so the judge wrapper sets `bypass_temperature=True`; the evaluator
  needs a generous `max_tokens` (faithfulness decomposes the answer into claims);
  and RAGAS 0.4.x imports a Vertex path LangChain 1.x removed, so one unused
  symbol is shimmed before import. See `ragas_eval.py`.
- **Observability vs. evaluation.** Evals (here) measure *quality*; Langfuse
  tracing measures *what happened at runtime* (latency, cost, token usage). Both
  are wired in — see [AGENT.md](AGENT.md) §7b.
