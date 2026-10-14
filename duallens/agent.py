"""DualLens Agent — an agentic, tool-using investment screener.

Where `DualLens_Analytics.ipynb` runs a fixed pipeline (pull metrics -> retrieve
-> rank), this turns the same two lenses into *tools* and lets Claude decide
what to call and when. Claude Opus 4.8 runs a manual tool-use loop over:

    get_stock_metrics   — quantitative lens (Yahoo Finance)
    get_price_performance — 3-yr momentum (Yahoo Finance)
    search_ai_strategy  — qualitative lens (RAG over the AI-initiative PDFs)

Framework-free on purpose: the agent loop is a plain `while` over the Messages
API, so every decision the model makes is explicit and inspectable — no
orchestration library hiding the control flow.

Run:
    python -m duallens.agent --ingest        # one-time: build the vector store
    python -m duallens.agent                 # default: rank all 5 companies
    python -m duallens.agent "Is NVIDIA a buy given its AI strategy?"

Needs ANTHROPIC_API_KEY (reasoning) and OPENAI_API_KEY (embeddings) in a .env
at the repo root. Claude has no embeddings API, so OpenAI handles embeddings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import anthropic
import chromadb
import openai
import yfinance as yf
from dotenv import load_dotenv
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PDF_DIR = REPO_ROOT / "Companies-AI-Initiatives"
CHROMA_DIR = HERE / "chroma_agent"

LLM_MODEL = "claude-opus-4-8"          # all reasoning (Opus 4.8: adaptive thinking, no temperature)
EMBED_MODEL = "text-embedding-3-small"  # embeddings only — Claude has none
COLLECTION_NAME = "duallens_agent"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4
MAX_TOKENS = 4096
MAX_TURNS = 12  # safety cap on the agent loop

COMPANIES = {
    "AMZN": "Amazon",
    "GOOGL": "Alphabet (Google)",
    "IBM": "IBM",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
}
NAME_TO_TICKER = {name: t for t, name in COMPANIES.items()}


def _load_keys() -> None:
    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    load_dotenv()  # also try CWD
    import os

    missing = [k for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if not os.getenv(k)]
    if missing:
        sys.exit(
            f"Missing {', '.join(missing)}. Copy .env.example to .env at the repo "
            "root and fill in your keys."
        )


# ---------------------------------------------------------------------------
# Embeddings + vector store (OpenAI embeddings, local Chroma)
# ---------------------------------------------------------------------------
def _openai_client() -> "openai.OpenAI":
    # OPENAI_BASE_URL lets you point at a proxy (e.g. a Great Learning gateway)
    # instead of api.openai.com. Unset -> the real OpenAI endpoint.
    import os

    base_url = os.getenv("OPENAI_BASE_URL") or None
    return openai.OpenAI(base_url=base_url)


def _embed(texts: List[str]) -> List[List[float]]:
    resp = _openai_client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = " ".join(text.split())
    chunks, start, step = [], 0, max(1, size - overlap)
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(piece)
        start += step
    return chunks


def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def ingest() -> None:
    """Read every PDF, chunk, embed with OpenAI, and persist to Chroma."""
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {PDF_DIR}")

    collection = _collection()
    ids, docs, metas = [], [], []
    for pdf in pdfs:
        name = COMPANIES.get(pdf.stem, pdf.stem)
        text = "\n".join((p.extract_text() or "") for p in PdfReader(str(pdf)).pages)
        for i, chunk in enumerate(_chunk(text)):
            ids.append(f"{pdf.stem}::{i}")
            docs.append(chunk)
            metas.append({"company": name, "ticker": pdf.stem, "chunk": i})

    # Embed in batches so a big corpus doesn't blow the request size.
    embeddings: List[List[float]] = []
    for i in range(0, len(docs), 128):
        embeddings.extend(_embed(docs[i : i + 128]))

    collection.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    print(f"Ingested {len(ids)} chunks from {len(pdfs)} PDFs into {CHROMA_DIR}")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------
def _resolve_tickers(companies: List[str] | None) -> List[str]:
    if not companies:
        return list(COMPANIES)
    out = []
    for c in companies:
        if c in COMPANIES:              # already a ticker
            out.append(c)
        elif c in NAME_TO_TICKER:       # a full name
            out.append(NAME_TO_TICKER[c])
        else:                            # loose match on name
            match = next((t for t, n in COMPANIES.items() if c.lower() in n.lower()), None)
            out.append(match or c)
    return out


def tool_get_stock_metrics(companies: List[str] | None = None) -> Dict:
    """Market Cap, P/E, Dividend Yield, Beta, Total Revenue per company."""
    fields = {
        "market_cap": "marketCap",
        "pe_ratio": "trailingPE",
        "dividend_yield": "dividendYield",
        "beta": "beta",
        "total_revenue": "totalRevenue",
    }
    result = {}
    for ticker in _resolve_tickers(companies):
        info = yf.Ticker(ticker).info
        result[COMPANIES.get(ticker, ticker)] = {
            label: info.get(key) for label, key in fields.items()
        }
    return result


def tool_get_price_performance(companies: List[str] | None = None, period: str = "3y") -> Dict:
    """3-yr price momentum: start/end price and total return %, normalised so
    companies are comparable regardless of share price."""
    result = {}
    for ticker in _resolve_tickers(companies):
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            result[COMPANIES.get(ticker, ticker)] = {"error": "no price data"}
            continue
        start, end = float(hist["Close"].iloc[0]), float(hist["Close"].iloc[-1])
        result[COMPANIES.get(ticker, ticker)] = {
            "period": period,
            "start_price": round(start, 2),
            "end_price": round(end, 2),
            "total_return_pct": round((end / start - 1) * 100, 1),
        }
    return result


def tool_search_ai_strategy(query: str, top_k: int = TOP_K) -> Dict:
    """RAG retrieval over the AI-initiative PDFs — the qualitative lens."""
    collection = _collection()
    if collection.count() == 0:
        return {"error": "Vector store is empty. Run `python -m duallens.agent --ingest` first."}
    q_emb = _embed([query])
    res = collection.query(query_embeddings=q_emb, n_results=top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return {
        "query": query,
        "passages": [
            {"company": m.get("company"), "text": d} for d, m in zip(docs, metas)
        ],
    }


TOOLS = [
    {
        "name": "get_stock_metrics",
        "description": (
            "Quantitative lens. Returns fundamental financial metrics (market cap, "
            "trailing P/E, dividend yield, beta, total revenue) from Yahoo Finance for "
            "the given companies. Omit `companies` to get all five."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "companies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tickers or names, e.g. ['NVDA','Microsoft']. Omit for all.",
                }
            },
        },
    },
    {
        "name": "get_price_performance",
        "description": (
            "Quantitative lens. Returns start price, end price, and total return % over "
            "a period (default 3y) — the momentum signal. Omit `companies` for all five."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "companies": {"type": "array", "items": {"type": "string"}},
                "period": {"type": "string", "description": "e.g. '1y', '3y', '5y'."},
            },
        },
    },
    {
        "name": "search_ai_strategy",
        "description": (
            "Qualitative lens. Semantic search over the companies' AI-initiative reports. "
            "Use it to ground any claim about a company's AI strategy or future readiness. "
            "Returns the most relevant passages with their source company."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for in the reports."},
                "top_k": {"type": "integer", "description": "How many passages (default 4)."},
            },
            "required": ["query"],
        },
    },
]

_DISPATCH = {
    "get_stock_metrics": tool_get_stock_metrics,
    "get_price_performance": tool_get_price_performance,
    "search_ai_strategy": tool_search_ai_strategy,
}

SYSTEM_PROMPT = (
    "You are DualLens, an investment-analysis agent that ranks companies by fusing two "
    "lenses: QUANTITATIVE financial performance and QUALITATIVE AI-strategy signal.\n\n"
    "You have tools for both lenses. Decide which to call and when — you don't have to "
    "call all of them. Rules:\n"
    "1. Ground every qualitative claim about a company's AI strategy in a `search_ai_strategy` "
    "result; never assert AI-strategy facts from memory.\n"
    "2. When you give an investment view, reference BOTH a number (from the quant tools) and a "
    "specific AI initiative (from retrieval) for each company you discuss.\n"
    "3. If the user asks for a ranking, end with a Buy/Hold/Sell call per company and one clearly "
    "labelled single best pick, with one-line reasoning each.\n"
    "Be concise and concrete. Cite the company a passage came from."
)


# ---------------------------------------------------------------------------
# Agent loop (manual Messages-API tool-use loop)
# ---------------------------------------------------------------------------
def _anthropic_client() -> "anthropic.Anthropic":
    # ANTHROPIC_BASE_URL -> point at a gateway/proxy (unset = api.anthropic.com).
    # ANTHROPIC_WORKSPACE_ID -> required when the key isn't scoped to a workspace;
    # sent as the anthropic-workspace-id header on every request.
    import os

    kwargs: Dict = {}
    if os.getenv("ANTHROPIC_BASE_URL"):
        kwargs["base_url"] = os.getenv("ANTHROPIC_BASE_URL")
    if os.getenv("ANTHROPIC_WORKSPACE_ID"):
        kwargs["default_headers"] = {"anthropic-workspace-id": os.getenv("ANTHROPIC_WORKSPACE_ID")}
    return anthropic.Anthropic(**kwargs)


def run_agent(question: str, verbose: bool = True) -> str:
    client = _anthropic_client()
    messages: List[Dict] = [{"role": "user", "content": question}]

    for turn in range(MAX_TURNS):
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Surface any text the model emitted this turn.
        if verbose:
            for block in resp.content:
                if block.type == "text" and block.text.strip():
                    print(block.text)

        if resp.stop_reason != "tool_use":
            final = "".join(b.text for b in resp.content if b.type == "text")
            return final.strip()

        # Echo the assistant turn back, then run each requested tool.
        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            if verbose:
                print(f"  → {block.name}({json.dumps(block.input)})")
            try:
                output = _DISPATCH[block.name](**block.input)
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(output, default=str),
                }
            except Exception as exc:  # return errors to the model, don't crash the loop
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": f"Tool error: {exc}",
                    "is_error": True,
                }
            tool_results.append(result_block)
        messages.append({"role": "user", "content": tool_results})

    return "Stopped: reached the maximum number of agent turns without a final answer."


DEFAULT_QUESTION = (
    "Rank Amazon, Alphabet, IBM, Microsoft, and NVIDIA for a long-term investor by fusing "
    "their financials with their AI strategy. Give a Buy/Hold/Sell call per company and name "
    "the single best pick."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="DualLens agentic investment screener.")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION, help="What to ask the agent.")
    parser.add_argument("--ingest", action="store_true", help="Build/refresh the vector store, then exit.")
    parser.add_argument("--quiet", action="store_true", help="Only print the final answer.")
    args = parser.parse_args()

    _load_keys()

    if args.ingest:
        ingest()
        return

    if _collection().count() == 0:
        print("Vector store is empty — ingesting the PDFs first...\n")
        ingest()
        print()

    answer = run_agent(args.question, verbose=not args.quiet)
    if args.quiet:
        print(answer)


if __name__ == "__main__":
    main()
