"""RAGAS evaluation of the DualLens agent — the standard RAG metric suite.

Runs the agent on a small gold set, captures what it retrieved and answered, and
scores it with RAGAS:

    faithfulness        — are the answer's claims supported by retrieved context?
                          (hallucination)                                [no label]
    answer_relevancy    — does the answer address the question?          [no label]
    context_precision   — were the retrieved passages relevant?          [no label]
    context_recall      — did retrieval cover what the reference needs?  [needs gold]
    conciseness         — rubric (AspectCritic): full but not padded (0/1) [no label]
    correctness_rubric  — graded 1-5 rubric vs the reference answer      [needs gold]

Claude Opus 4.8 is the evaluator LLM and OpenAI embeddings back answer_relevancy —
the same clients the rest of the project uses (gateway + workspace headers honoured).

    python -m duallens.ragas_eval            # full gold set
    python -m duallens.ragas_eval -n 2       # first N questions (cheaper)

Note: RAGAS 0.4.x imports a Vertex AI path that LangChain 1.x removed, so we shim
that one unused symbol before importing ragas (see `_shim_vertexai`). We never use
Vertex; it only needs the name to exist at import time.
"""
from __future__ import annotations

import argparse
import sys
import types
from typing import Dict, List

from duallens import agent
from duallens.pipeline import config as pconfig


def _shim_vertexai() -> None:
    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.ChatVertexAI = type("ChatVertexAI", (), {})  # stub; unused
    sys.modules[name] = mod


# Gold set: AI-strategy questions with a concise reference answer grounded in the
# PDFs. The reference is only used by context_recall / correctness-style metrics.
GOLD = [
    {
        "question": "Which company's AI strategy centers on GPUs and the CUDA software ecosystem?",
        "reference": "NVIDIA — its moat is GPUs plus the CUDA software ecosystem (with Omniverse and AI microservices), making it the backbone of modern AI compute.",
    },
    {
        "question": "What is Alphabet's flagship enterprise AI platform and its multi-agent protocol?",
        "reference": "Alphabet's Vertex AI platform, together with the Agent2Agent (A2A) protocol for multi-agent orchestration.",
    },
    {
        "question": "What is Amazon's generative-AI platform on AWS?",
        "reference": "Amazon Bedrock (alongside SageMaker) is Amazon's generative-AI platform on AWS.",
    },
    {
        "question": "What does IBM's AI strategy focus on?",
        "reference": "IBM focuses on AI governance and security — watsonx Governance and Guardium AI Security for responsible, enterprise-grade AI.",
    },
]


def _run_and_capture(question: str) -> Dict:
    """Run the agent; collect the passages it retrieved + its final answer."""
    contexts: List[str] = []

    def on_event(ev: Dict) -> None:
        if ev["type"] == "tool_result" and ev.get("name") == "search_ai_strategy" and "output" in ev:
            for p in ev["output"].get("passages", []):
                contexts.append(f"[{p.get('company')}] {p.get('text')}")

    answer = agent.run_agent(question, verbose=False, on_event=on_event)
    return {"answer": answer, "contexts": contexts}


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS eval for the DualLens agent.")
    parser.add_argument("-n", type=int, default=len(GOLD), help="Evaluate the first N gold questions.")
    args = parser.parse_args()

    agent._load_keys()
    if agent._collection().count() == 0:
        print("Vector store empty — ingesting first...\n")
        agent.ingest()

    _shim_vertexai()
    import warnings

    warnings.filterwarnings("ignore")
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        AspectCritic,
        Faithfulness,
        LLMContextPrecisionWithoutReference,
        LLMContextRecall,
        ResponseRelevancy,
        RubricsScore,
    )

    # Opus 4.8 rejects `temperature`; RAGAS sets it unless we bypass it.
    # max_tokens must be generous — faithfulness decomposes the answer into claims
    # and can run long; too small -> LLMDidNotFinishException -> NaN.
    evaluator_llm = LangchainLLMWrapper(pconfig.get_llm(max_tokens=4096))
    evaluator_llm.bypass_temperature = True
    evaluator_llm.bypass_n = True  # Anthropic has no `n`; replicate prompts instead
    evaluator_emb = LangchainEmbeddingsWrapper(pconfig.get_embeddings())

    # Conciseness as a rubric (binary): rewards a full but non-padded answer.
    conciseness = AspectCritic(
        name="conciseness",
        definition=(
            "Return 1 if the answer is concise — it answers fully without unnecessary "
            "repetition, filler, or padding. Return 0 if it is bloated or rambling."
        ),
    )

    # A graded 1-5 correctness rubric scored against the reference answer.
    correctness_rubric = RubricsScore(
        name="correctness_rubric",
        rubrics={
            "score1_description": "Factually wrong or contradicts the reference.",
            "score2_description": "Mostly incorrect; only a minor detail matches the reference.",
            "score3_description": "Partially correct; captures some of the reference but misses or muddles key points.",
            "score4_description": "Largely correct and aligned with the reference, with a small omission or imprecision.",
            "score5_description": "Fully correct and complete with respect to the reference.",
        },
    )

    questions = GOLD[: args.n]
    print(f"Running the agent on {len(questions)} gold question(s), then scoring with RAGAS...\n")

    samples = []
    for i, item in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {item['question']}")
        run = _run_and_capture(item["question"])
        print(f"    retrieved {len(run['contexts'])} passage(s)")
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                retrieved_contexts=run["contexts"] or ["(no passages retrieved)"],
                response=run["answer"],
                reference=item["reference"],
            )
        )

    print("\nScoring (this makes several judge calls per metric)...\n")
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithoutReference(),
            LLMContextRecall(),
            conciseness,
            correctness_rubric,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_emb,
        show_progress=True,
    )

    print("\n" + "=" * 60)
    print("RAGAS SCORECARD (0-1, higher is better)")
    print("=" * 60)
    print(result)

    df = result.to_pandas()
    out = pconfig.PIPELINE_DIR.parent / "ragas_results.csv"
    df.to_csv(out, index=False)
    print(f"\nPer-question detail written to {out}")


if __name__ == "__main__":
    main()
