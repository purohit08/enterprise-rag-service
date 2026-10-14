"""Central configuration for the PDF RAG agent."""
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "Companies-AI-Initiatives"   # your 5 company PDFs live here
CHROMA_DIR = BASE_DIR / "chroma_store"            # persistent vector store

# --- Hugging Face models ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"            # small, fast, high-quality embeddings
GEN_MODEL = "Qwen/Qwen2.5-3B-Instruct"            # ungated instruct LLM, runs on Apple MPS

# --- Chunking (character-based) ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- Retrieval / generation ---
TOP_K = 4
MAX_NEW_TOKENS = 512

# --- Chroma ---
COLLECTION_NAME = "pdf_chunks"
