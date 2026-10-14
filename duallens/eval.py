"""Evaluate the DualLens agent with an LLM-as-judge.

For each question in an eval set, this runs the agent, captures the evidence it
actually gathered (retrieved passages + the numeric tool outputs), and asks a
judge LLM to score the final answer 1-5 on:

    groundedness — is every claim supported by the evidence the agent gathered?
    relevance    — does the answer actually address the question?

It prints a per-question scorecard and the averages, and writes the full results
to duallens/eval_results.json.

    python -m duallens.eval                 # run the built-in question set
    python -m duallens.eval -q "Is IBM a buy?" "Rank all five by momentum"
    python -m duallens.eval --json results.json

Why an LLM judge? It is cheap, fast, and repeatable for regression testing — run
it after changing the prompt, chunk size, or top_k to see if quality moved. Bias
is mitigated with a clear rubric and by grading only against gathered evidence.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List

from duallens import agent

HERE = Path(__file__).resolve().parent

# A small, fixed eval set. Swap in a labelled gold set to track quality over time.
EVAL_QUESTIONS = [
    "Which company is investing most heavily in generative AI, and how?",
    "Compare NVIDIA and IBM for a risk-averse income investor.",
    "Is Microsoft overvalued relative to its AI strategy?",
    "Rank all five companies by 3-year momentum and name the leader.",
    "Which company has the strongest enterprise-AI governance story?",
]

JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator of a financial-analysis agent. You are given a "
    "QUESTION, the EVIDENCE the agent gathered (retrieved document passages and numeric tool "
    "outputs), and the agent's ANSWER. Score the answer from 1 (poor) to 5 (excellent) on:\n"
    "- groundedness: is every factual claim supported by the EVIDENCE? Unsupported or invented "
    "claims lower this hard.\n"
    "- relevance: does the answer directly address the QUESTION?\n"
    'Return ONLY JSON: {"groundedness": int, "relevance": int, "rationale": "one sentence"}'
)


def _run_and_capture(question: str) -> Dict:
    """Run the agent and collect the evidence it gathered plus the final answer."""
    evidence: List[str] = []
    tools_used: List[str] = []

    def on_event(ev: Dict) -> None:
        if ev["type"] == "tool":
            tools_used.append(ev["name"])
        elif ev["type"] == "tool_result" and "output" in ev:
            out = ev["output"]
            if ev["name"] == "search_ai_strategy":
                for p in out.get("passages", []):
                    evidence.append(f"[{p.get('company')}] {p.get('text')}")
            else:  # metrics / price tools -> compact JSON
                evidence.append(f"[{ev['name']}] {json.dumps(out, default=str)}")

    answer = agent.run_agent(question, verbose=False, on_event=on_event)
    return {
        "question": question,
        "answer": answer,
        "evidence": "\n\n".join(evidence),
        "tools_used": tools_used,
    }


def _extract_json(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {"error": text}
    except json.JSONDecodeError:
        return {"error": text}


def _judge(client, run: Dict) -> Dict:
    user = (
        f"QUESTION:\n{run['question']}\n\n"
        f"EVIDENCE:\n{run['evidence'] or '(none gathered)'}\n\n"
        f"ANSWER:\n{run['answer']}"
    )
    resp = client.messages.create(
        model=agent.LLM_MODEL,
        max_tokens=500,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return _extract_json(text)


def evaluate(questions: List[str], out_path: Path) -> None:
    agent._load_keys()
    if agent._collection().count() == 0:
        print("Vector store empty — ingesting first...\n")
        agent.ingest()

    client = agent._anthropic_client()
    results = []

    print(f"Evaluating {len(questions)} question(s) with LLM-as-judge ({agent.LLM_MODEL})\n")
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q}")
        run = _run_and_capture(q)
        scores = _judge(client, run)
        g, r = scores.get("groundedness"), scores.get("relevance")
        print(f"    groundedness={g}  relevance={r}  — {scores.get('rationale', scores.get('error', ''))}")
        print(f"    tools: {', '.join(run['tools_used']) or '(none)'}\n")
        results.append({**run, "scores": scores})

    # Aggregate
    grounded = [r["scores"]["groundedness"] for r in results if isinstance(r["scores"].get("groundedness"), int)]
    relevant = [r["scores"]["relevance"] for r in results if isinstance(r["scores"].get("relevance"), int)]

    print("=" * 56)
    print("SCORECARD (1-5)")
    if grounded:
        print(f"  Avg groundedness: {statistics.mean(grounded):.2f}  (n={len(grounded)})")
    if relevant:
        print(f"  Avg relevance:    {statistics.mean(relevant):.2f}  (n={len(relevant)})")
    print("=" * 56)

    out_path.write_text(json.dumps({"results": results}, indent=2, default=str))
    print(f"\nFull results written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge eval for the DualLens agent.")
    parser.add_argument("-q", "--questions", nargs="+", help="Questions to evaluate (default: built-in set).")
    parser.add_argument("--json", default=str(HERE / "eval_results.json"), help="Where to write full results.")
    args = parser.parse_args()
    evaluate(args.questions or EVAL_QUESTIONS, Path(args.json))


if __name__ == "__main__":
    main()
