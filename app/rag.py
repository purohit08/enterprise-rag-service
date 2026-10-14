"""RAG core: PDF -> chunk -> embed -> Chroma -> retrieve -> generate.

Everything runs locally. Heavy models (embedder, LLM) are loaded lazily on
first use so the FastAPI server starts instantly.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app import config


# ---------------------------------------------------------------------------
# Device detection (Apple Silicon -> MPS)
# ---------------------------------------------------------------------------
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = get_device()


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBED_MODEL, device=DEVICE)


@functools.lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


@functools.lru_cache(maxsize=1)
def get_generator():
    """Load the instruct LLM (tokenizer + model). ~6 GB download on first run."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.GEN_MODEL,
        dtype=torch.float16,
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


# ---------------------------------------------------------------------------
# PDF reading + chunking
# ---------------------------------------------------------------------------
def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Sliding-window character chunks with overlap."""
    text = " ".join(text.split())  # normalize whitespace
    chunks, start, n = [], 0, len(text)
    step = max(1, size - overlap)
    while start < n:
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def ingest_pdfs(pdf_dir: Optional[Path] = None) -> Dict:
    pdf_dir = Path(pdf_dir) if pdf_dir else config.PDF_DIR
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        return {"status": "no_pdfs", "pdf_dir": str(pdf_dir), "chunks": 0}

    embedder = get_embedder()
    collection = get_collection()

    ids, texts, metas = [], [], []
    per_file = {}
    for path in pdf_paths:
        chunks = chunk_text(read_pdf(path), config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        per_file[path.name] = len(chunks)
        for i, chunk in enumerate(chunks):
            ids.append(f"{path.name}::{i}")
            texts.append(chunk)
            metas.append({"source": path.name, "chunk": i})

    embeddings = embedder.encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True
    ).tolist()

    # upsert => re-ingesting is idempotent (same ids overwrite)
    collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)

    return {
        "status": "ok",
        "files": per_file,
        "total_chunks": len(ids),
        "pdf_dir": str(pdf_dir),
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(question: str, top_k: Optional[int] = None) -> List[Dict]:
    top_k = top_k or config.TOP_K
    embedder = get_embedder()
    collection = get_collection()

    q_emb = embedder.encode([question], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=top_k)

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    return [
        {
            "text": d,
            "source": m.get("source"),
            "chunk": m.get("chunk"),
            "distance": dist,
        }
        for d, m, dist in zip(docs, metas, dists)
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about companies' AI "
    "initiatives. Use ONLY the provided context. If the answer is not in the "
    "context, say you don't know. Cite sources inline like [1], [2]."
)


def _build_messages(question: str, contexts: List[Dict]) -> List[Dict]:
    context_block = "\n\n".join(
        f"[{i}] (from {c['source']}) {c['text']}" for i, c in enumerate(contexts, 1)
    )
    user = f"Context:\n{context_block}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate_answer(
    question: str, contexts: List[Dict], max_new_tokens: Optional[int] = None
) -> str:
    max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS
    tokenizer, model = get_generator()
    messages = _build_messages(question, contexts)

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    generated = output[0][prompt_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
def answer_question(
    question: str, top_k: Optional[int] = None, generate: bool = True
) -> Dict:
    contexts = retrieve(question, top_k)
    result: Dict = {"question": question, "sources": contexts}
    if not contexts:
        result["answer"] = "No relevant content found — have you ingested the PDFs?"
    elif generate:
        result["answer"] = generate_answer(question, contexts)
    else:
        result["answer"] = None
    return result
