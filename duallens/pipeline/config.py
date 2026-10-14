"""Shared configuration and LangChain client factories for the pipeline.

Keeps all tunables (models, chunking, paths) and the two LLM/embedding clients in
one place. The clients read optional gateway/workspace settings from the
environment so the same code runs against OpenAI/Anthropic directly or through a
proxy (see the repo-root `.env`).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# --- Paths ---
PIPELINE_DIR = Path(__file__).resolve().parent
DUALLENS_DIR = PIPELINE_DIR.parent
REPO_ROOT = DUALLENS_DIR.parent
PDF_DIR = REPO_ROOT / "Companies-AI-Initiatives"
CHROMA_DIR = PIPELINE_DIR / "chroma_pipeline"

# --- Models (Claude reasons; OpenAI embeds — Claude has no embeddings API) ---
LLM_MODEL = "claude-opus-4-8"
EMBED_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "duallens_pipeline"

# --- Chunking / retrieval ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
MAX_TOKENS = 1500

COMPANIES = {
    "AMZN": "Amazon",
    "GOOGL": "Alphabet (Google)",
    "IBM": "IBM",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
}


def load_keys() -> None:
    """Load API keys from the repo-root .env (and CWD) and verify they exist."""
    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    load_dotenv()
    missing = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if not os.getenv(k)]
    if missing:
        sys.exit(
            f"Missing {', '.join(missing)}. Copy .env.example to .env at the repo root "
            "and fill in your keys."
        )


def get_llm(max_tokens: int = MAX_TOKENS):
    """ChatAnthropic for Claude Opus 4.8. No temperature — Opus 4.8 rejects it.

    Honours ANTHROPIC_BASE_URL (gateway) and ANTHROPIC_WORKSPACE_ID (header for
    keys that aren't scoped to a workspace).
    """
    from langchain_anthropic import ChatAnthropic

    kwargs = {}
    if os.getenv("ANTHROPIC_BASE_URL"):
        kwargs["anthropic_api_url"] = os.getenv("ANTHROPIC_BASE_URL")
    if os.getenv("ANTHROPIC_WORKSPACE_ID"):
        kwargs["default_headers"] = {"anthropic-workspace-id": os.getenv("ANTHROPIC_WORKSPACE_ID")}
    return ChatAnthropic(model=LLM_MODEL, max_tokens=max_tokens, **kwargs)


def get_embeddings():
    """OpenAIEmbeddings. Honours OPENAI_BASE_URL (OpenAI-compatible gateway)."""
    from langchain_openai import OpenAIEmbeddings

    kwargs = {}
    if os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
    return OpenAIEmbeddings(model=EMBED_MODEL, **kwargs)
