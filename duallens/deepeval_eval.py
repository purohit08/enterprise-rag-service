"""DeepEval evaluation of the DualLens agent — the Eval Frameworks option.

DeepEval (confident-ai/deepeval) is a pytest-style LLM eval framework. This runs
the agent on the gold set and scores it with DeepEval's RAG metrics plus two
GEval rubric metrics, using Claude Opus 4.8 as the judge (via a custom model
wrapper so the workspace header / gateway are honoured — no OpenAI judge needed).

    faithfulness        — claims supported by retrieval_context (hallucination)
    answer_relevancy     — does the answer address the question?
    contextual_precision — were retrieved passages relevant? (needs expected)
    contextual_recall    — did retrieval cover the expected answer? (needs expected)
    conciseness (GEval)  — rubric: full but not padded
    correctness (GEval)  — rubric: matches the expected answer

    python -m duallens.deepeval_eval          # full gold set
    python -m duallens.deepeval_eval -n 2     # first N (cheaper)

Complements `ragas_eval.py` (RAGAS) and `eval.py` (custom judge) — same gold set,
a different, widely-used framework.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from typing import Dict, List

# DeepEval phones home / checks for updates by default — turn that off.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")

from duallens import agent
from duallens.ragas_eval import GOLD, _run_and_capture  # reuse gold set + capture


def _extract_json(text: str) -> Dict:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    match = re.search(r"[\{\[].*[\}\]]", text, re.DOTALL)
    return json.loads(match.group(0)) if match else json.loads(text)


def _build_claude_model():
    """A DeepEval custom LLM backed by Claude Opus 4.8 (our configured client)."""
    from deepeval.models.base_model import DeepEvalBaseLLM

    class ClaudeModel(DeepEvalBaseLLM):
        def __init__(self, model: str = agent.LLM_MODEL, max_tokens: int = 4096):
            self._client = agent._anthropic_client()
            self._model = model
            self.max_tokens = max_tokens

        def load_model(self):
            return self._client

        def get_model_name(self) -> str:
            return self._model

        def generate(self, prompt: str, schema=None):
            content = prompt
            if schema is not None:
                content += (
                    "\n\nRespond with ONLY a JSON object that conforms to this JSON "
                    f"schema (no prose, no code fences):\n{json.dumps(schema.model_json_schema())}"
                )
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": content}],
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            if schema is not None:
                return schema.model_validate(_extract_json(text))
            return text

        async def a_generate(self, prompt: str, schema=None):
            import asyncio

            return await asyncio.to_thread(self.generate, prompt, schema)

    return ClaudeModel()


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepEval eval for the DualLens agent.")
    parser.add_argument("-n", type=int, default=len(GOLD), help="Evaluate the first N gold questions.")
    args = parser.parse_args()

    agent._load_keys()
    if agent._collection().count() == 0:
        print("Vector store empty — ingesting first...\n")
        agent.ingest()

    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    judge = _build_claude_model()
    P = LLMTestCaseParams

    def metrics():
        return [
            FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True),
            AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True),
            ContextualPrecisionMetric(threshold=0.7, model=judge, include_reason=True),
            ContextualRecallMetric(threshold=0.7, model=judge, include_reason=True),
            GEval(
                name="conciseness",
                model=judge,
                criteria="Is the answer concise — fully answering without unnecessary repetition, filler, or padding?",
                evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT],
            ),
            GEval(
                name="correctness",
                model=judge,
                criteria="Is the actual output factually correct and complete with respect to the expected output?",
                evaluation_params=[P.INPUT, P.ACTUAL_OUTPUT, P.EXPECTED_OUTPUT],
            ),
        ]

    questions = GOLD[: args.n]
    print(f"Running the agent on {len(questions)} gold question(s), then scoring with DeepEval...\n")

    scores: Dict[str, List[float]] = {}
    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {item['question']}")
        run = _run_and_capture(item["question"])
        print(f"    retrieved {len(run['contexts'])} passage(s)")
        tc = LLMTestCase(
            input=item["question"],
            actual_output=run["answer"],
            expected_output=item["reference"],
            retrieval_context=run["contexts"] or ["(no passages retrieved)"],
        )
        for metric in metrics():
            try:
                metric.measure(tc)
                scores.setdefault(metric.__class__.__name__ if not getattr(metric, "name", None) else metric.name, []).append(metric.score)
                print(f"    {getattr(metric, 'name', metric.__class__.__name__):22s} {metric.score:.3f}")
            except Exception as exc:
                print(f"    {getattr(metric, 'name', metric.__class__.__name__):22s} ERROR: {exc}")
        print()

    print("=" * 60)
    print("DEEPEVAL SCORECARD (0-1, higher is better)")
    print("=" * 60)
    for name, vals in scores.items():
        if vals:
            print(f"  {name:24s} {statistics.mean(vals):.3f}  (n={len(vals)})")
    print("=" * 60)


if __name__ == "__main__":
    main()
