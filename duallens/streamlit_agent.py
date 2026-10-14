"""Web UI for the DualLens agent — runs in the browser via Streamlit.

    streamlit run duallens/streamlit_agent.py

Opens at http://localhost:8501. Shows the agent's live tool-call trace as it
fuses the quantitative and qualitative lenses, then the final ranking.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Make `from duallens import agent` work when run via `streamlit run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from duallens import agent  # noqa: E402

st.set_page_config(page_title="DualLens Agent", page_icon="🔭", layout="centered")

st.title("🔭 DualLens Agent")
st.caption(
    "An agentic investment screener. Claude Opus 4.8 decides which tools to call — "
    "financial metrics, price momentum, and RAG over AI-initiative reports — then "
    "fuses the quantitative and qualitative lenses into a grounded ranking."
)


@st.cache_resource(show_spinner=False)
def _prepare() -> dict:
    """Load keys once and ensure the vector store is populated."""
    agent._load_keys()
    collection = agent._collection()
    if collection.count() == 0:
        agent.ingest()
        collection = agent._collection()
    return {
        "chunks": collection.count(),
        "model": agent.LLM_MODEL,
        "embed": agent.EMBED_MODEL,
        "traced": bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")),
    }


# --- Sidebar: status + example questions ---
with st.sidebar:
    st.header("Status")
    try:
        info = _prepare()
        st.success("Ready")
        st.metric("Chunks indexed", info["chunks"])
        st.write(f"**Reasoning:** `{info['model']}`")
        st.write(f"**Embeddings:** `{info['embed']}`")
        st.write(f"**Tracing:** {'Langfuse ✅' if info['traced'] else 'off'}")
    except SystemExit as exc:
        st.error(str(exc))
        st.stop()

    st.header("Try")
    examples = [
        agent.DEFAULT_QUESTION,
        "Is NVIDIA a buy given its AI strategy?",
        "Compare IBM and Microsoft for a risk-averse income investor.",
        "Which company has the strongest generative-AI platform strategy?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex


# --- Main: question + run ---
question = st.text_area(
    "Your question",
    value=st.session_state.get("question", agent.DEFAULT_QUESTION),
    height=100,
)

if st.button("Run agent", type="primary"):
    trace = st.container()
    trace.subheader("Agent trace")
    answer_box = st.empty()

    TOOL_LABELS = {
        "get_stock_metrics": "📊 Financial metrics",
        "get_price_performance": "📈 Price performance",
        "search_ai_strategy": "🔎 AI-strategy retrieval",
    }

    final_text = {"value": ""}

    def on_event(ev: dict) -> None:
        if ev["type"] == "text" and ev["text"].strip():
            trace.markdown(f"💭 {ev['text']}")
        elif ev["type"] == "tool":
            label = TOOL_LABELS.get(ev["name"], ev["name"])
            trace.markdown(f"**{label}** — `{ev['input']}`")
        elif ev["type"] == "final":
            final_text["value"] = ev["text"]

    with st.spinner("Thinking and calling tools…"):
        try:
            result = agent.run_agent(question, verbose=False, on_event=on_event)
        except Exception as exc:  # surface API/tool errors in the UI
            st.error(f"Run failed: {exc}")
            st.stop()

    st.subheader("Answer")
    st.markdown(final_text["value"] or result)

st.divider()
st.caption("Analysis only — not personalised financial advice.")
