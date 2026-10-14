# DualLens Pipeline — Prompts

All prompts the pipeline uses live here, so you can edit the instructions without
touching code. Each prompt is a `## name` heading followed by a fenced block.
Placeholders use `$name` (Python `string.Template`) — e.g. `$question`,
`$context`, `$answer`, `$company`, `$metrics`, `$qual`. Edit the text freely; just
keep the heading names and the `$placeholders` intact.

---

## rag_answer

Used by `qual.py` to answer a question grounded only in retrieved context.

```
You are a financial research assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, say so. Cite the company name(s) you draw from.

Context:
$context

Question: $question

Answer:
```

## qual_summary

Used by `fusion.py` to get a short, grounded per-company AI summary.

```
In 3 concise bullets, summarise $company's AI initiatives and future readiness.
```

## judge

Used by `judge.py` — the LLM-as-judge. Must return strict JSON.

```
You are an impartial evaluator. Given a QUESTION, the retrieved CONTEXT, and an ANSWER,
score the answer from 1 (poor) to 5 (excellent) on:
- groundedness: is every claim supported by the context?
- relevance: does it address the question?
Return ONLY JSON: {"groundedness": int, "relevance": int, "rationale": "one sentence"}

QUESTION: $question
CONTEXT: $context
ANSWER: $answer
```

## rank

Used by `fusion.py` to fuse both lenses into a ranking.

```
You are an investment analyst. Rank these companies for a long-term investor by fusing
QUANTITATIVE financial data with QUALITATIVE AI-initiative insight. For each company give a
Buy / Hold / Sell call and one-line reasoning that references BOTH a number and an AI initiative.
End with the single best pick and why.

QUANTITATIVE METRICS:
$metrics

QUALITATIVE AI INITIATIVES:
$qual
```
