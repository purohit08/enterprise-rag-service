"""Loads the prompts from `prompts.md` so instructions live in one editable file.

Each prompt in prompts.md is a `## name` heading followed by a fenced block.
`render(name, **vars)` substitutes `$placeholders` (Python string.Template).
"""
from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Dict

PROMPTS_FILE = Path(__file__).resolve().parent / "prompts.md"

_BLOCK = re.compile(r"^##\s+([a-z_]+)\s*\n.*?\n```[a-zA-Z]*\n(.*?)\n```", re.MULTILINE | re.DOTALL)


def load_prompts() -> Dict[str, str]:
    text = PROMPTS_FILE.read_text(encoding="utf-8")
    return {m.group(1): m.group(2).strip() for m in _BLOCK.finditer(text)}


PROMPTS = load_prompts()


def render(name: str, **variables: str) -> str:
    """Return prompt `name` with its $placeholders filled in."""
    if name not in PROMPTS:
        raise KeyError(f"Prompt '{name}' not found in {PROMPTS_FILE.name}. Have: {list(PROMPTS)}")
    return string.Template(PROMPTS[name]).safe_substitute(**variables)
