"""Run the whole DualLens pipeline top-to-bottom — the notebook as a script.

    python -m duallens.pipeline.run

Stages (each lives in its own module):
    quant  -> qual (+ judge) -> fusion
"""
from __future__ import annotations

import json

from duallens.pipeline import config, fusion, judge, qual, quant


def _section(title: str) -> None:
    print("\n" + "=" * 68 + f"\n{title}\n" + "=" * 68)


def main() -> None:
    config.load_keys()
    llm = config.get_llm()

    # --- §2 Quantitative lens ---
    _section("§2  QUANTITATIVE LENS — financials & momentum")
    perf = quant.price_performance()
    for name, p in perf.items():
        print(f"  {name:22s} 3y return: {p.get('total_return_pct')}%")
    metrics_str = quant.metrics_table_str()
    print("\n" + metrics_str)

    # --- §3 Qualitative lens (RAG) ---
    _section("§3  QUALITATIVE LENS — RAG over AI-initiative PDFs")
    store = qual.build_vectorstore()
    retriever = qual.get_retriever(store)
    sample_q = "Which companies are investing most heavily in generative AI, and how?"
    print(f"Q: {sample_q}\n")
    print(qual.answer(sample_q, retriever, llm))

    # --- §3d Evaluation (LLM-as-judge) ---
    _section("§3d  EVALUATION — LLM-as-judge (groundedness + relevance)")
    verdict = judge.evaluate(sample_q, retriever, llm)
    print("Scores:", json.dumps(verdict["scores"]))

    # --- §4 Fusion (ranking) ---
    _section("§4  FUSION — ranking (quant + qual)")
    print("Collecting per-company qualitative summaries...")
    summaries = fusion.qualitative_summaries(retriever, llm)
    print()
    print(fusion.rank(metrics_str, summaries, llm))


if __name__ == "__main__":
    main()
