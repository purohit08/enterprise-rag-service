"""§3 Qualitative lens — RAG over the AI-initiative PDFs (LangChain).

Mirrors notebook cells 3a-3c: load & chunk the PDFs, embed & store in Chroma,
then retrieval-augmented answering grounded strictly in the retrieved context.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from duallens.pipeline import config, prompts


def build_vectorstore(force: bool = False) -> Chroma:
    """Load & chunk the PDFs, embed with OpenAI, persist to Chroma. Idempotent:
    reuses an existing store unless `force=True`."""
    embeddings = config.get_embeddings()
    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )
    if store._collection.count() > 0 and not force:
        return store

    docs = PyPDFDirectoryLoader(str(config.PDF_DIR)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} pages -> {len(chunks)} chunks; embedding & indexing...")
    store.add_documents(chunks)
    return store


def get_retriever(store: Chroma, k: int = config.TOP_K):
    return store.as_retriever(search_kwargs={"k": k})


def format_docs(retrieved) -> str:
    """Tag each passage with its source company, as the RAG prompt expects."""
    return "\n\n".join(
        f"[{Path(d.metadata.get('source', '?')).stem}] {d.page_content}" for d in retrieved
    )


def answer(question: str, retriever, llm) -> str:
    """Retrieve, then let Claude answer grounded in the retrieved context."""
    context = format_docs(retriever.invoke(question))
    prompt = prompts.render("rag_answer", context=context, question=question)
    return llm.invoke(prompt).content
