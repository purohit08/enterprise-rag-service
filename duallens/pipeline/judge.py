"""§3d Evaluation — LLM-as-judge (groundedness + relevance).

Mirrors notebook cell 3d: score a RAG answer 1-5 on whether every claim is
supported by the retrieved context (groundedness) and whether it addresses the
question (relevance). Returns parsed JSON.
"""
from __future__ import annotations

import json
import re
from typing import Dict

from duallens.pipeline import prompts, qual


def _extract_json(text: str) -> Dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(match.group(0)) if match else {"error": text}
    except json.JSONDecodeError:
        return {"error": text}


def evaluate(question: str, retriever, llm) -> Dict:
    """Answer the question, then judge the answer against its retrieved context."""
    context = qual.format_docs(retriever.invoke(question))
    answer = qual.answer(question, retriever, llm)
    verdict = llm.invoke(
        prompts.render("judge", question=question, context=context, answer=answer)
    ).content
    return {"answer": answer, "scores": _extract_json(verdict)}
