"""§4 Fusion — scoring & ranking (quant + qual).

Mirrors notebook cells 19-20: collect a grounded per-company qualitative summary,
then give Claude both lenses (the metrics table + the summaries) and have it rank
the companies with Buy/Hold/Sell calls and a single best pick.
"""
from __future__ import annotations

from typing import Dict

from duallens.pipeline import config, prompts, qual


def qualitative_summaries(retriever, llm) -> Dict[str, str]:
    """One short, PDF-grounded AI summary per company."""
    summaries = {}
    for name in config.COMPANIES.values():
        q = prompts.render("qual_summary", company=name)
        summaries[name] = qual.answer(q, retriever, llm)
    return summaries


def rank(metrics_str: str, qual_summaries: Dict[str, str], llm) -> str:
    """Fuse the quantitative table and qualitative summaries into a ranking."""
    qual_block = "\n\n".join(f"### {name}\n{text}" for name, text in qual_summaries.items())
    prompt = prompts.render("rank", metrics=metrics_str, qual=qual_block)
    return llm.invoke(prompt).content
